# B1 实际耗时（2026-10-25T03:44:14+08:00，A10，128 envs）
- prior: 83.6s（290/1000 epochs，validation_plateau 早停）
- SL 19: 见 records/b1-sl.json（runtime 字段）
- warmup 32: 171.6s（每 update ≈4.2s，含编译开销）
- selfplay 256: 1007.1s（每 update ≈3.9s，单块完成未分块）
- 导出+validate+4 对战: 约 25min（03:34–03:59）
- B1 总 GPU 消耗 ≈ 0.6h，远低于 14.4h 预算（A10 实测速度远超规格保守估计）
