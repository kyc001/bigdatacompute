# Final Run Freeze - 2026-04-27

## 冻结参数

- beta: 0.85
- eps: 1e-8
- mode: csr_block
- K: 8
- dtype: float32
- max_iter: 200

## 运行机器信息

- platform.platform(): `Windows-10-10.0.26200-SP0`
- sys.version: `3.9.23 | packaged by conda-forge | (main, Jun  4 2025, 17:49:16) [MSC v.1929 64 bit (AMD64)]`
- numpy.__version__: `2.0.2`
- scipy.__version__: `1.13.1`

## stdout JSON

```json
{"peak_rss_mb": 29.578125, "wall_sec": 0.8275939999999999, "iters": 15, "mode": "csr_block", "K": 8, "dtype": "float32", "delta": 7.170456228777766e-09, "n_nodes": 10000, "n_edges": 150000, "top10_signature": "75,8686,9678,5104,725,3257,468,7730,7175,5526"}
```

## top10_signature 金标准

`75,8686,9678,5104,725,3257,468,7730,7175,5526`
