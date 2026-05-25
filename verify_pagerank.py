"""使用 networkx 独立验证 PageRank 结果的正确性。

用法:
    .venv_verify/bin/python verify_pagerank.py [Data.txt路径] [Res.txt路径]

验证层次:
    1. 基本图统计信息
    2. networkx PageRank 与 Res.txt 的 Top-100 比较
    3. 排名一致性与分值精度检查
"""

import sys
import os

import numpy as np
import networkx as nx

# Fix encoding for Windows GBK terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def load_edges(path: str) -> list[tuple[int, int]]:
    edges = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = line.split()
            edges.append((int(u), int(v)))
    return edges


def load_res(path: str) -> list[tuple[int, float]]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            data.append((int(parts[0]), float(parts[1])))
    return data


def verify(data_path: str, res_path: str, beta: float = 0.85,
           eps: float = 1e-8, max_iter: int = 200) -> dict:
    print("=" * 65)
    print("  PageRank 结果独立验证 (networkx 作为参考实现)")
    print("=" * 65)

    # ── Step 1: 加载数据 ──────────────────────────────────
    print("\n[Step 1] 加载数据")
    edges = load_edges(data_path)
    print(f"  Data.txt: {len(edges)} 条边")

    # 确定节点 ID 范围，与项目保持一致：当 min=0 且密度 > 50% 时
    # 项目保持完整 ID 空间 0..max_id（包含未在边中出现的孤立节点）
    all_ids = set()
    for u, v in edges:
        all_ids.add(u)
        all_ids.add(v)
    min_id = min(all_ids)
    max_id = max(all_ids)
    seen_count = len(all_ids)
    span = max_id - min_id + 1
    density = seen_count / span

    if min_id == 0 and density >= 0.5:
        n_nodes = span  # 保持完整 ID 空间
        print(f"  ID 空间: 0..{max_id}, 出现 {seen_count} 个节点, 密度 {density:.1%}")
        print(f"  使用完整 ID 空间: {n_nodes} 个节点 (含 {n_nodes - seen_count} 个孤立节点)")
    else:
        n_nodes = seen_count
        print(f"  节点数: {n_nodes}")

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))  # 确保所有节点都在图中
    G.add_edges_from(edges)
    n_edges = G.number_of_edges()
    dead_ends = [n for n in G.nodes() if G.out_degree(n) == 0]
    print(f"  边数: {n_edges}")
    print(f"  Dead-end 节点数: {len(dead_ends)} ({100*len(dead_ends)/n_nodes:.1f}%)")

    # ── Step 2: networkx PageRank ──────────────────────────
    print(f"\n[Step 2] networkx.pagerank() 计算")
    print(f"  参数: alpha={beta}, tol={eps}, max_iter={max_iter}")
    print(f"  公式: r = (1-alpha)/N + alpha*dangling_mass/N + alpha * M^T r")

    pr = nx.pagerank(
        G, alpha=beta, tol=eps, max_iter=max_iter, weight=None
    )
    pr_sum = sum(pr.values())
    print(f"  PageRank 总和: {pr_sum:.10f}")

    top100_nx = sorted(pr.items(), key=lambda x: (-x[1], x[0]))[:100]

    print(f"\n  networkx Top 10:")
    for i, (nid, score) in enumerate(top100_nx[:10]):
        print(f"    {i+1:2d}. Node {nid:>6d}: {score:.10f}")

    # ── Step 3: 读取 Res.txt ───────────────────────────────
    print(f"\n[Step 3] 读取 Res.txt")
    res_data = load_res(res_path)
    print(f"  记录数: {len(res_data)}")

    print(f"\n  Res.txt Top 10:")
    for i, (nid, score) in enumerate(res_data[:10]):
        print(f"    {i+1:2d}. Node {nid:>6d}: {score:.10f}")

    # ── Step 4: 对比分析 ───────────────────────────────────
    print(f"\n[Step 4] 对比分析")

    # 建立 networkx 查找表
    nx_dict = dict(pr)

    # 4a. Top-100 节点重叠
    res_ids = {nid for nid, _ in res_data}
    nx_ids = {nid for nid, _ in top100_nx}
    overlap = res_ids & nx_ids
    missing_from_nx = res_ids - nx_ids
    missing_from_res = nx_ids - res_ids
    print(f"\n  Top-100 节点重叠: {len(overlap)} / 100")
    if missing_from_nx:
        print(f"  仅在 Res.txt 中的节点 (不在 nx top-100):")
        for nid in sorted(missing_from_nx):
            res_score = next(s for i, s in res_data if i == nid)
            nx_score = nx_dict.get(nid, 0)
            print(f"    Node {nid}: Res={res_score:.10f}, nx={nx_score:.10f}")

    if missing_from_res:
        print(f"  仅在 networkx top-100 中的节点 (不在 Res.txt):")
        for nid in sorted(missing_from_res):
            nx_score = nx_dict[nid]
            print(f"    Node {nid}: nx={nx_score:.10f}")

    # 4b. 逐节点分差
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    diffs = []
    for nid, res_score in res_data:
        nx_score = nx_dict.get(nid, 0.0)
        abs_diff = abs(res_score - nx_score)
        rel_diff = abs_diff / max(nx_score, 1e-15) if nx_score > 1e-15 else abs_diff
        diffs.append((nid, res_score, nx_score, abs_diff, rel_diff))
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)

    print(f"\n  分值差异 (Top-100 共同节点):")
    print(f"    最大绝对差: {max_abs_diff:.2e}")
    print(f"    最大相对差: {max_rel_diff:.2e}")

    # 4c. 排名位置偏移
    res_rank = {nid: i for i, (nid, _) in enumerate(res_data)}
    nx_rank = {nid: i for i, (nid, _) in enumerate(top100_nx)}
    rank_shifts = []
    for nid in overlap:
        rank_shifts.append(abs(res_rank[nid] - nx_rank[nid]))

    if rank_shifts:
        avg_shift = np.mean(rank_shifts)
        max_shift = max(rank_shifts)
        print(f"\n  排名位置偏移 (Top-100 共同节点):")
        print(f"    平均偏移: {avg_shift:.2f} 位")
        print(f"    最大偏移: {max_shift} 位")
        # 偏移分布
        bins = [0, 0.5, 1.5, 5.5, 20.5, 100]
        labels = ["0位", "1位", "2-5位", "6-20位", "21+位"]
        for i in range(len(bins) - 1):
            count = sum(1 for s in rank_shifts if bins[i] <= s <= bins[i+1])
            print(f"    {labels[i]}: {count}")

    # 4d. 整体分数分布对比
    print(f"\n  整体分数分布:")
    res_scores = [s for _, s in res_data]
    nx_top_scores = [s for _, s in top100_nx]
    print(f"    Res.txt : min={min(res_scores):.2e}, max={max(res_scores):.2e}, "
          f"avg={np.mean(res_scores):.2e}")
    print(f"    nx      : min={min(nx_top_scores):.2e}, max={max(nx_top_scores):.2e}, "
          f"avg={np.mean(nx_top_scores):.2e}")

    # 4e. 全局所有节点的排名一致性
    # 对所有节点比较排名
    all_sorted_res = sorted(res_data, key=lambda x: (-x[1], x[0]))
    all_sorted_nx = sorted(pr.items(), key=lambda x: (-x[1], x[0]))

    # Kendall Tau-b correlation on rankings (同序对数)
    res_global_rank = {nid: i for i, (nid, _) in enumerate(all_sorted_res)}
    nx_global_rank = {nid: i for i, (nid, _) in enumerate(all_sorted_nx)}
    common_nodes = set(res_global_rank.keys()) & set(nx_global_rank.keys())

    concordant = 0
    discordant = 0
    common_list = sorted(common_nodes)
    for i in range(len(common_list)):
        for j in range(i + 1, len(common_list)):
            a, b = common_list[i], common_list[j]
            res_order = res_global_rank[a] - res_global_rank[b]
            nx_order = nx_global_rank[a] - nx_global_rank[b]
            if res_order * nx_order > 0:
                concordant += 1
            elif res_order * nx_order < 0:
                discordant += 1

    total_pairs = concordant + discordant
    if total_pairs > 0:
        kendall_tau = (concordant - discordant) / total_pairs
        print(f"\n  全局排名一致性:")
        print(f"    共享节点数: {len(common_nodes)}")
        print(f"    Kendall Tau-b: {kendall_tau:.6f} (1.0 = 完全一致)")

    # ── Step 5: 结论 ───────────────────────────────────────
    print(f"\n[Step 5] 验证结论")
    print("-" * 65)

    all_ok = True

    # 检查1: 概率和
    if abs(pr_sum - 1.0) < 1e-9:
        print(f"  [OK] PageRank 概率和为 1.0 ({pr_sum:.10f})")
    else:
        print(f"  [FAIL] PageRank 概率和偏离 1.0 ({pr_sum:.10f})")
        all_ok = False

    # 检查2: Top-100 重叠
    if len(overlap) >= 98:
        print(f"  [OK] Top-100 重叠率优秀 ({len(overlap)}/100)")
    elif len(overlap) >= 95:
        print(f"  ~ Top-100 重叠率良好 ({len(overlap)}/100)")
    else:
        print(f"  [FAIL] Top-100 重叠率偏低 ({len(overlap)}/100)")
        all_ok = False

    # 检查3: 分值精度
    if max_rel_diff < 1e-6:
        print(f"  [OK] 分值精度优秀 (max_rel_diff={max_rel_diff:.2e})")
    elif max_rel_diff < 1e-4:
        print(f"  ~ 分值精度良好 (max_rel_diff={max_rel_diff:.2e})，"
              f"可能是 float32/float64 差异")
    else:
        print(f"  [FAIL] 分值精度不足 (max_rel_diff={max_rel_diff:.2e})")
        all_ok = False

    # 检查4: 排名一致性
    if total_pairs > 0:
        if kendall_tau > 0.999:
            print(f"  [OK] 全局排名高度一致 (Kendall Tau={kendall_tau:.6f})")
        elif kendall_tau > 0.99:
            print(f"  ~ 全局排名基本一致 (Kendall Tau={kendall_tau:.6f})")
        else:
            print(f"  [FAIL] 全局排名存在较大差异 (Kendall Tau={kendall_tau:.6f})")
            all_ok = False

    print("-" * 65)
    if all_ok:
        print("  最终结论: Res.txt 与独立 reference 实现一致 [OK]")
    else:
        print("  最终结论: 存在差异，需要进一步排查 [FAIL]")

    return {
        "overlap": len(overlap),
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "kendall_tau": kendall_tau if total_pairs > 0 else None,
        "all_ok": all_ok,
    }


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "Data.txt"
    res_path = sys.argv[2] if len(sys.argv) > 2 else "Res.txt"
    verify(data_path, res_path)
