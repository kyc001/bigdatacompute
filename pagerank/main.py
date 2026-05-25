"""PageRank 主入口。

算法侧职责：
- `load_graph`：读取 `Data.txt`，构造按源点分组的 CSR；
- `power_iteration`：纯 CSR 幂迭代，统一使用 dead-end 质量补偿；
- `dump_top_k`：按冻结接口输出 Top-k，并恢复原始 NodeID；
- CLI：同时兼容 `dense / csr / block / csr_block` 四种模式。
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
from array import array
from typing import Iterable, Iterator

import numpy as np



_INT32_MAX = np.iinfo(np.int32).max
_IDENTITY_DENSITY_THRESHOLD = 0.5
_RUNTIME_TMP_ROOT = ".tmp_runtime"

# 由于冻结接口没有显式暴露 id_map，这里缓存最近一次 load_graph 的反向映射，
# 供 dump_top_k 在写 Res.txt 时把内部编号还原成原始 NodeID。
_LAST_NODE_IDS: np.ndarray | None = None


def get_base_dir() -> str:
    """源码运行时返回 main.py 所在目录；exe 运行时返回 main.exe 所在目录。"""

    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(filename: str) -> str:
    """返回 PyInstaller 内嵌资源路径；源码运行时返回程序同目录路径。"""

    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(get_base_dir(), filename)


def default_data_path() -> str:
    """优先读取程序同目录 Data.txt；没有时读取 PyInstaller 内嵌 Data.txt。"""

    external_path = os.path.join(get_base_dir(), "Data.txt")
    if os.path.exists(external_path):
        return external_path

    embedded_path = resource_path("Data.txt")
    if os.path.exists(embedded_path):
        return embedded_path

    return external_path


def default_out_path() -> str:
    """Res.txt 始终写到程序同目录。"""

    return os.path.join(get_base_dir(), "Res.txt")


# 每条边记录 8 字节：src(int32) + local_dst(int32)。
BLOCK_RECORD_DTYPE = np.dtype([("src", np.int32), ("local_dst", np.int32)])
BLOCK_RECORD_STRUCT = struct.Struct("<ii")


def iter_edges_from_csr(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
) -> tuple[Iterator[tuple[int, int]], int]:
    """把 CSR 视图转换为边流，供 build_blocks 直接消费。

    时间复杂度：O(M)
    空间复杂度：O(1) 额外空间（不把整张图重新展开到内存）
    """

    if row_ptr.ndim != 1 or col_idx.ndim != 1:
        raise ValueError("row_ptr 与 col_idx 必须是一维数组")
    if row_ptr.size == 0:
        raise ValueError("row_ptr 不能为空")

    n_nodes = int(row_ptr.size - 1)

    def edge_iter() -> Iterator[tuple[int, int]]:
        for src in range(n_nodes):
            start = int(row_ptr[src])
            stop = int(row_ptr[src + 1])
            for offset in range(start, stop):
                yield src, int(col_idx[offset])

    return edge_iter(), n_nodes


def _normalize_edges_input(
    edges,
) -> tuple[Iterable[tuple[int, int]], int]:
    """统一解析 build_blocks 的输入。

    支持：
    - `(edge_iterable, N)`：推荐，用于主流程；
    - `np.ndarray(M, 2)`：用于 mock / tests。
    """

    if isinstance(edges, tuple) and len(edges) == 2:
        edge_iterable, n_nodes = edges
        n_nodes = int(n_nodes)
        if n_nodes <= 0:
            raise ValueError("N 必须为正整数")
        return edge_iterable, n_nodes

    if isinstance(edges, np.ndarray):
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges 数组必须是形如 (M, 2) 的二维数组")
        if edges.size == 0:
            raise ValueError("edges 不能为空")
        n_nodes = int(edges.max()) + 1
        return ((int(src), int(dst)) for src, dst in edges), n_nodes

    raise ValueError("edges 必须是 (edge_iterable, N) 或 np.ndarray(M, 2)")


def build_blocks(edges, K: int, tmp_dir: str) -> list[dict]:
    """按目标节点分桶，把边写入 K 个二进制临时文件。

    设计意图：
    - 用目标节点分块，而不是源节点分块；
    - 每个块单独落盘，后续可由 np.memmap 逐块读取；
    - 文件中存储的是 `(src, local_dst)`，避免迭代阶段重复做全局下标减法。

    时间复杂度：O(M)
    空间复杂度：O(K)，不依赖边数 M 的额外常驻内存
    """

    if K < 1:
        raise ValueError("K 必须大于等于 1")

    edge_iterable, n_nodes = _normalize_edges_input(edges)
    os.makedirs(tmp_dir, exist_ok=True)

    block_paths: list[str] = []
    block_files = []
    edge_counts = np.zeros(K, dtype=np.int32)
    block_starts = np.empty(K, dtype=np.int32)
    block_stops = np.empty(K, dtype=np.int32)
    node_to_block = np.empty(n_nodes, dtype=np.int32)

    for block_id in range(K):
        block_starts[block_id] = (block_id * n_nodes) // K
        block_stops[block_id] = ((block_id + 1) * n_nodes) // K
        node_to_block[block_starts[block_id] : block_stops[block_id]] = block_id

    for block_id in range(K):
        path = os.path.join(tmp_dir, f"block_{block_id:03d}.bin")
        block_paths.append(path)
        block_files.append(open(path, "wb"))

    try:
        for src, dst in edge_iterable:
            if src < 0 or dst < 0 or dst >= n_nodes or src >= n_nodes:
                raise ValueError("检测到非法节点编号，必须落在 [0, N) 区间")

            block_id = int(node_to_block[dst])
            node_start = int(block_starts[block_id])
            local_dst = dst - node_start
            block_files[block_id].write(BLOCK_RECORD_STRUCT.pack(src, local_dst))
            edge_counts[block_id] += 1
    finally:
        for file_obj in block_files:
            file_obj.close()

    metadata: list[dict] = []
    for block_id, path in enumerate(block_paths):
        node_start = int(block_starts[block_id])
        node_stop = int(block_stops[block_id])
        metadata.append(
            {
                "block_id": int(block_id),
                "path": path,
                "node_start": int(node_start),
                "node_stop": int(node_stop),
                "node_count": int(node_stop - node_start),
                "edge_count": int(edge_counts[block_id]),
            }
        )
    return metadata


def iterate_by_block(
    blocks: list[dict],
    out_deg: np.ndarray,
    N: int,
    beta: float = 0.85,
    eps: float = 1e-8,
    dtype: np.dtype = np.float32,
    max_iter: int = 200,
    tmp_dir: str | None = None,
    chunk_size: int = 1 << 15,
) -> tuple[np.ndarray, int, float]:
    """使用磁盘映射和真分块执行 PageRank 幂迭代。

    核心策略：
    - 完整 `r` 与 `r_new` 都放到 memmap 文件中；
    - 每轮只取当前块的 `r_new[node_start:node_stop]` 视图做更新；
    - 块内边记录通过 memmap 顺序读取，避免把整张反向邻接表留在内存。

    时间复杂度：O(T * M)，T 为收敛轮数
    空间复杂度：O(N + max_block_edges_chunk)，其中 `r_new` 的活动工作集仅为当前块片段
    """

    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype 仅支持 np.float32 或 np.float64")
    if N <= 0:
        raise ValueError("N 必须为正整数")
    if out_deg.shape[0] != N:
        raise ValueError("out_deg 长度必须等于 N")
    if not blocks:
        raise ValueError("blocks 不能为空")

    own_tmp_dir = False
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="pagerank_rank_")
        own_tmp_dir = True
    else:
        os.makedirs(tmp_dir, exist_ok=True)

    rank_a_path = os.path.join(tmp_dir, f"rank_a_{dtype.__name__}.bin")
    rank_b_path = os.path.join(tmp_dir, f"rank_b_{dtype.__name__}.bin")

    live_mask = out_deg > 0
    dead_mask = ~live_mask
    inv_out_deg = np.zeros(N, dtype=dtype)
    inv_out_deg[live_mask] = 1.0 / out_deg[live_mask]
    max_block_nodes = max(int(block["node_count"]) for block in blocks)

    src_rank_buffer = np.empty(chunk_size, dtype=dtype)
    contrib_buffer = np.empty(chunk_size, dtype=dtype)
    delta_buffer = np.empty(max_block_nodes, dtype=dtype)

    r = None
    r_new = None
    try:
        r = np.memmap(rank_a_path, dtype=dtype, mode="w+", shape=(N,))
        r_new = np.memmap(rank_b_path, dtype=dtype, mode="w+", shape=(N,))
        r.fill(dtype(1.0 / N))

        last_delta = np.inf
        for num_iter in range(1, max_iter + 1):
            dangling_mass = float(np.sum(r[dead_mask], dtype=np.float64))
            base = dtype((1.0 - beta) / N + beta * dangling_mass / N)
            l1_delta = 0.0

            for block in blocks:
                node_start = int(block["node_start"])
                node_stop = int(block["node_stop"])
                node_count = int(block["node_count"])
                edge_count = int(block["edge_count"])

                block_view = r_new[node_start:node_stop]
                block_view.fill(base)

                if edge_count > 0:
                    edge_map = np.memmap(
                        block["path"],
                        dtype=BLOCK_RECORD_DTYPE,
                        mode="r",
                        shape=(edge_count,),
                    )
                    src_idx = None
                    local_dst = None
                    try:
                        for offset in range(0, edge_count, chunk_size):
                            chunk_stop = min(offset + chunk_size, edge_count)
                            current = chunk_stop - offset
                            src_idx = edge_map["src"][offset:chunk_stop]
                            local_dst = edge_map["local_dst"][offset:chunk_stop]

                            # 先把 rank[src] 和 1/out_deg[src] 拉到固定缓冲区，再原地乘起来。
                            np.take(r, src_idx, out=src_rank_buffer[:current])
                            np.take(inv_out_deg, src_idx, out=contrib_buffer[:current])
                            np.multiply(
                                src_rank_buffer[:current],
                                contrib_buffer[:current],
                                out=contrib_buffer[:current],
                            )
                            contrib_buffer[:current] *= beta
                            np.add.at(block_view, local_dst, contrib_buffer[:current])
                    finally:
                        del src_idx
                        del local_dst
                        mmap_obj = getattr(edge_map, "_mmap", None)
                        if mmap_obj is not None:
                            mmap_obj.close()
                        del edge_map

                # 按块累计 L1 范数，避免一次性构造完整差值向量。
                block_delta = delta_buffer[:node_count]
                block_delta[:] = block_view
                block_delta -= r[node_start:node_stop]
                np.abs(block_delta, out=block_delta)
                l1_delta += float(np.sum(block_delta, dtype=np.float64))
                del block_delta
                del block_view

            last_delta = l1_delta
            if l1_delta < eps:
                result = np.array(r_new, dtype=dtype, copy=True)
                return result, num_iter, last_delta

            r, r_new = r_new, r

        result = np.array(r, dtype=dtype, copy=True)
        return result, max_iter, float(last_delta)
    finally:
        if r is not None:
            try:
                r.flush()
            except ValueError:
                pass
            mmap_obj = getattr(r, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        if r_new is not None:
            try:
                r_new.flush()
            except ValueError:
                pass
            mmap_obj = getattr(r_new, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        del r
        del r_new
        gc.collect()

        for path in (rank_a_path, rank_b_path):
            if os.path.exists(path):
                os.remove(path)
        if own_tmp_dir and os.path.isdir(tmp_dir):
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass


if os.name == "nt":
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
    _KERNEL32.GetCurrentProcess.restype = ctypes.c_void_p

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    _PSAPI.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    _PSAPI.GetProcessMemoryInfo.restype = ctypes.c_int


def _normalize_dtype(dtype: np.dtype | type[np.floating]) -> np.dtype:
    """把 dtype 统一规整成 numpy dtype。"""

    normalized = np.dtype(dtype)
    if normalized not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype 仅支持 float32 或 float64")
    return normalized


def _get_process_rss_bytes() -> int:
    """读取当前进程 RSS，优先返回真实工作集大小。"""

    if os.name == "nt":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        handle = _KERNEL32.GetCurrentProcess()
        success = _PSAPI.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if success:
            return int(counters.WorkingSetSize)
        return 0

    try:
        import resource
    except ImportError:
        return 0

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return 0
    if os.uname().sysname == "Darwin":
        return int(usage)
    return int(usage * 1024)


class PeakRssSampler:
    """后台轮询当前进程 RSS 峰值。"""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.peak_rss = _get_process_rss_bytes()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def _worker(self) -> None:
        while not self._stop_event.wait(self.interval):
            self.peak_rss = max(self.peak_rss, _get_process_rss_bytes())

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.peak_rss = max(self.peak_rss, _get_process_rss_bytes())


def _parse_edge_line(line: str, line_no: int) -> tuple[int, int]:
    """解析一行边记录。"""

    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"第 {line_no} 行格式错误：应为两个整数")

    try:
        src = int(parts[0])
        dst = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"第 {line_no} 行格式错误：存在非整数节点编号") from exc

    if src < 0 or dst < 0:
        raise ValueError(f"第 {line_no} 行格式错误：节点编号必须为非负整数")
    if src > _INT32_MAX or dst > _INT32_MAX:
        raise ValueError(f"第 {line_no} 行格式错误：节点编号超出 int32 可表示范围")
    return src, dst


def _should_keep_identity_ids(min_node_id: int, max_node_id: int, seen_count: int) -> bool:
    """决定是否保留 0..max 的原始编号空间。

    这样做是为了不丢掉“出现在编号范围内、但在边表中完全缺席”的孤立点。
    若编号并非从 0 起步，或编号跨度远大于已观测节点数，则退化为紧凑重映射。
    """

    if seen_count == 0 or min_node_id != 0:
        return False
    span = max_node_id + 1
    density = seen_count / float(span)
    return density >= _IDENTITY_DENSITY_THRESHOLD


def _build_node_mapping(raw_src: np.ndarray, raw_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据原始节点编号构造内部编号与反向映射。"""

    seen_ids = set(map(int, raw_src.tolist()))
    seen_ids.update(map(int, raw_dst.tolist()))
    if not seen_ids:
        raise ValueError("输入文件为空")

    min_node_id = min(seen_ids)
    max_node_id = max(seen_ids)
    if _should_keep_identity_ids(min_node_id, max_node_id, len(seen_ids)):
        node_ids = np.arange(max_node_id + 1, dtype=np.int64)
        src_idx = raw_src.astype(np.int32, copy=False)
        dst_idx = raw_dst.astype(np.int32, copy=False)
        return src_idx, dst_idx, node_ids

    node_ids = np.array(sorted(seen_ids), dtype=np.int64)
    node_to_idx = {int(node_id): idx for idx, node_id in enumerate(node_ids.tolist())}
    src_idx = np.empty(raw_src.shape[0], dtype=np.int32)
    dst_idx = np.empty(raw_dst.shape[0], dtype=np.int32)
    for edge_index in range(raw_src.shape[0]):
        src_idx[edge_index] = node_to_idx[int(raw_src[edge_index])]
        dst_idx[edge_index] = node_to_idx[int(raw_dst[edge_index])]
    return src_idx, dst_idx, node_ids


def _build_csr_from_edges(src_idx: np.ndarray, dst_idx: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把边列表压缩成按源点分组的 CSR。"""

    out_deg = np.zeros(n_nodes, dtype=np.int32)
    for src in src_idx:
        out_deg[int(src)] += 1

    row_ptr = np.empty(n_nodes + 1, dtype=np.int32)
    row_ptr[0] = 0
    edge_cursor = 0
    for node in range(n_nodes):
        edge_cursor += int(out_deg[node])
        if edge_cursor > _INT32_MAX:
            raise ValueError("边数超过 int32 可表示范围，无法构造 CSR")
        row_ptr[node + 1] = edge_cursor

    col_idx = np.empty(src_idx.shape[0], dtype=np.int32)
    write_ptr = row_ptr[:-1].copy()
    for edge_index in range(src_idx.shape[0]):
        src = int(src_idx[edge_index])
        pos = int(write_ptr[src])
        col_idx[pos] = dst_idx[edge_index]
        write_ptr[src] = pos + 1

    del write_ptr
    return row_ptr, col_idx, out_deg


def load_graph(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """读取 Data.txt 并构造 CSR。

    返回：
        row_ptr: np.ndarray[np.int32], shape = (N + 1,)
        col_idx: np.ndarray[np.int32], shape = (M,)
        out_deg: np.ndarray[np.int32], shape = (N,)
        N: int
    """

    global _LAST_NODE_IDS
    _LAST_NODE_IDS = None

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    raw_src_buffer = array("I")
    raw_dst_buffer = array("I")
    has_edge = False

    with open(path, "r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"第 {line_no} 行为空，输入格式非法")

            src, dst = _parse_edge_line(line, line_no)
            raw_src_buffer.append(src)
            raw_dst_buffer.append(dst)
            has_edge = True

    if not has_edge:
        raise ValueError("输入文件为空")

    raw_src = np.frombuffer(raw_src_buffer, dtype=np.uint32).view(np.int32)
    raw_dst = np.frombuffer(raw_dst_buffer, dtype=np.uint32).view(np.int32)
    src_idx, dst_idx, node_ids = _build_node_mapping(raw_src, raw_dst)
    n_nodes = int(node_ids.shape[0])
    row_ptr, col_idx, out_deg = _build_csr_from_edges(src_idx, dst_idx, n_nodes)

    _LAST_NODE_IDS = node_ids

    del raw_src
    del raw_dst
    del src_idx
    del dst_idx
    del raw_src_buffer
    del raw_dst_buffer
    gc.collect()

    return row_ptr, col_idx, out_deg, n_nodes


def _validate_csr(row_ptr: np.ndarray, col_idx: np.ndarray, out_deg: np.ndarray, n_nodes: int) -> None:
    """校验 CSR 形状与 dtype。"""

    if n_nodes <= 0:
        raise ValueError("N 必须为正整数")
    if row_ptr.dtype != np.int32 or col_idx.dtype != np.int32 or out_deg.dtype != np.int32:
        raise ValueError("row_ptr / col_idx / out_deg 必须全部为 np.int32")
    if row_ptr.ndim != 1 or col_idx.ndim != 1 or out_deg.ndim != 1:
        raise ValueError("row_ptr / col_idx / out_deg 必须全部为一维数组")
    if row_ptr.shape[0] != n_nodes + 1:
        raise ValueError("row_ptr 长度必须为 N + 1")
    if out_deg.shape[0] != n_nodes:
        raise ValueError("out_deg 长度必须为 N")
    if int(row_ptr[-1]) != int(col_idx.shape[0]):
        raise ValueError("row_ptr[-1] 必须等于边数 M")
    for node in range(n_nodes):
        if int(row_ptr[node + 1]) - int(row_ptr[node]) != int(out_deg[node]):
            raise ValueError("out_deg 与 row_ptr 定义不一致")


def _finalize_ranks(ranks: np.ndarray) -> np.ndarray:
    """在返回前做轻量归一化，控制 float32 漂移。"""

    total = float(np.sum(ranks, dtype=np.float64))
    if total > 0.0 and abs(total - 1.0) > 1e-6:
        ranks /= ranks.dtype.type(total)
    return ranks


def _make_runtime_tmp_dir(prefix: str) -> str:
    """在工作区内创建可写的临时目录。

    这里不用系统默认临时目录，避免受沙箱或课程机权限策略影响。
    """

    os.makedirs(_RUNTIME_TMP_ROOT, exist_ok=True)
    tmp_dir = os.path.join(_RUNTIME_TMP_ROOT, f"{prefix}{os.getpid()}_{time.time_ns()}")
    os.makedirs(tmp_dir, exist_ok=False)
    return tmp_dir


def dense_power_iteration(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    out_deg: np.ndarray,
    N: int,
    beta: float,
    eps: float,
    dtype: np.dtype = np.float32,
    max_iter: int = 200,
) -> tuple[np.ndarray, int, float]:
    """稠密基线幂迭代，只用于 E1 对照实验。"""

    _validate_csr(row_ptr, col_idx, out_deg, N)
    dtype = _normalize_dtype(dtype)
    if max_iter < 1:
        raise ValueError("max_iter 必须大于等于 1")

    transition = np.zeros((N, N), dtype=dtype)
    for src in range(N):
        degree = int(out_deg[src])
        if degree == 0:
            continue
        start = int(row_ptr[src])
        stop = int(row_ptr[src + 1])
        transition[src, col_idx[start:stop]] = dtype.type(1.0 / degree)

    r = np.full(N, dtype.type(1.0 / N), dtype=dtype)
    r_new = np.empty(N, dtype=dtype)
    temp = np.empty(N, dtype=dtype)
    delta_buffer = np.empty(N, dtype=dtype)
    dead_mask = out_deg == 0
    base_teleport = dtype.type((1.0 - beta) / N)
    last_delta = float("inf")

    for num_iter in range(1, max_iter + 1):
        dangling_mass = float(np.sum(r[dead_mask], dtype=np.float64))
        base = dtype.type(base_teleport + beta * dangling_mass / N)
        np.dot(transition.T, r, out=temp)
        temp *= dtype.type(beta)
        r_new[:] = temp
        r_new += base

        delta_buffer[:] = r_new
        delta_buffer -= r
        np.abs(delta_buffer, out=delta_buffer)
        last_delta = float(np.sum(delta_buffer, dtype=np.float64))
        if last_delta < eps:
            return _finalize_ranks(r_new.copy()), num_iter, last_delta

        r, r_new = r_new, r

    return _finalize_ranks(r.copy()), max_iter, last_delta


def power_iteration(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    out_deg: np.ndarray,
    N: int,
    beta: float,
    eps: float,
    dtype: np.dtype = np.float32,
    max_iter: int = 200,
) -> tuple[np.ndarray, int, float]:
    """纯 CSR 幂迭代。

    更新公式固定为：

        r_new[v] = (1 - beta) / N
                 + beta * dangling_mass / N
                 + beta * sum(r[u] / out_deg[u] for u -> v)
    """

    _validate_csr(row_ptr, col_idx, out_deg, N)
    dtype = _normalize_dtype(dtype)
    if max_iter < 1:
        raise ValueError("max_iter 必须大于等于 1")

    r = np.full(N, dtype.type(1.0 / N), dtype=dtype)
    r_new = np.empty(N, dtype=dtype)
    delta_buffer = np.empty(N, dtype=dtype)

    live_mask = out_deg > 0
    dead_mask = ~live_mask
    live_nodes = np.flatnonzero(live_mask).astype(np.int32, copy=False)
    inv_out_deg = np.zeros(N, dtype=dtype)
    inv_out_deg[live_mask] = dtype.type(1.0) / out_deg[live_mask]
    beta_cast = dtype.type(beta)
    base_teleport = dtype.type((1.0 - beta) / N)
    last_delta = float("inf")

    for num_iter in range(1, max_iter + 1):
        dangling_mass = float(np.sum(r[dead_mask], dtype=np.float64))
        base = dtype.type(base_teleport + beta * dangling_mass / N)
        r_new.fill(base)

        for src in live_nodes:
            src_index = int(src)
            contrib = beta_cast * r[src_index] * inv_out_deg[src_index]
            start = int(row_ptr[src_index])
            stop = int(row_ptr[src_index + 1])
            np.add.at(r_new, col_idx[start:stop], contrib)

        delta_buffer[:] = r_new
        delta_buffer -= r
        np.abs(delta_buffer, out=delta_buffer)
        last_delta = float(np.sum(delta_buffer, dtype=np.float64))
        if last_delta < eps:
            return _finalize_ranks(r_new.copy()), num_iter, last_delta

        r, r_new = r_new, r

    return _finalize_ranks(r.copy()), max_iter, last_delta


def dump_top_k(path: str, ranks: np.ndarray, k: int = 100) -> str:
    """把 Top-k 写入 Res.txt，并返回 top10_signature。"""

    if ranks.ndim != 1:
        raise ValueError("ranks 必须为一维数组")
    if k < 1:
        raise ValueError("k 必须大于等于 1")

    total = int(ranks.shape[0])
    top_k = min(k, total)
    node_ids = _LAST_NODE_IDS
    if node_ids is None or int(node_ids.shape[0]) != total:
        node_ids = np.arange(total, dtype=np.int64)

    order = np.lexsort((node_ids, -ranks))
    top_idx = order[:top_k]

    with open(path, "w", encoding="utf-8") as file_obj:
        for internal_id in top_idx:
            original_id = int(node_ids[int(internal_id)])
            score = float(ranks[int(internal_id)])
            file_obj.write(f"{original_id} {score:.10f}\n")

    top10_ids = [str(int(node_ids[int(internal_id)])) for internal_id in top_idx[:10]]
    return ",".join(top10_ids)


def run_selected_mode(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    out_deg: np.ndarray,
    n_nodes: int,
    mode: str,
    K: int,
    beta: float,
    eps: float,
    dtype: np.dtype,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    """按 mode 分发到对应实现。"""

    if mode == "dense":
        return dense_power_iteration(
            row_ptr,
            col_idx,
            out_deg,
            n_nodes,
            beta=beta,
            eps=eps,
            dtype=dtype,
            max_iter=max_iter,
        )

    if mode == "csr":
        return power_iteration(
            row_ptr,
            col_idx,
            out_deg,
            n_nodes,
            beta=beta,
            eps=eps,
            dtype=dtype,
            max_iter=max_iter,
        )

    if mode in ("block", "csr_block"):
        if K < 1:
            raise ValueError("K 必须大于等于 1")
        tmp_dir = _make_runtime_tmp_dir("pagerank_blocks_")
        try:
            block_meta = build_blocks(iter_edges_from_csr(row_ptr, col_idx), K, tmp_dir)
            return iterate_by_block(
                block_meta,
                out_deg,
                n_nodes,
                beta=beta,
                eps=eps,
                dtype=dtype.type,
                max_iter=max_iter,
                tmp_dir=tmp_dir,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    raise ValueError("mode 仅支持 dense / csr / block / csr_block")


def run_pagerank(
    data_path: str,
    out_path: str,
    mode: str = "csr_block",
    K: int = 8,
    beta: float = 0.85,
    eps: float = 1e-8,
    dtype_name: str = "float32",
    max_iter: int = 200,
) -> dict[str, object]:
    """执行一次 PageRank，并返回 stdout JSON 摘要。"""

    dtype = _normalize_dtype(np.float32 if dtype_name == "float32" else np.float64)
    sampler = PeakRssSampler(interval=0.05)
    wall_start = time.perf_counter()
    wall_sec = 0.0
    n_nodes = 0
    n_edges = 0
    iters = 0
    delta = float("inf")
    top10_signature = ""

    sampler.start()
    try:
        row_ptr, col_idx, out_deg, n_nodes = load_graph(data_path)
        n_edges = int(col_idx.shape[0])
        ranks, iters, delta = run_selected_mode(
            row_ptr,
            col_idx,
            out_deg,
            n_nodes,
            mode=mode,
            K=K,
            beta=beta,
            eps=eps,
            dtype=dtype,
            max_iter=max_iter,
        )
        top10_signature = dump_top_k(out_path, ranks, k=100)
        wall_sec = time.perf_counter() - wall_start
    finally:
        sampler.stop()

    return {
        "peak_rss_mb": float(sampler.peak_rss / (1024 * 1024)),
        "wall_sec": float(wall_sec),
        "iters": int(iters),
        "mode": mode,
        "K": int(K),
        "dtype": dtype_name,
        "delta": float(delta),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "top10_signature": top10_signature,
    }


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="PageRank main entry")
    parser.add_argument("--data", default=default_data_path(), help="输入 Data.txt 路径")
    parser.add_argument("--beta", type=float, default=0.85, help="阻尼系数")
    parser.add_argument("--eps", type=float, default=1e-8, help="收敛阈值")
    parser.add_argument("--out", default=default_out_path(), help="输出 Res.txt 路径")
    parser.add_argument("--mode", choices=("dense", "csr", "block", "csr_block"), default="csr_block")
    parser.add_argument("--K", type=int, default=8, help="分块数量")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    summary = run_pagerank(
        data_path=args.data,
        out_path=args.out,
        mode=args.mode,
        K=args.K,
        beta=args.beta,
        eps=args.eps,
        dtype_name=args.dtype,
        max_iter=args.max_iter,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
