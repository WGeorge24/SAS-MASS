# SAS

本文方法的训练入口，包名 `SAS`，论文中的方法名为 SAS-MASS。在仓库根目录运行，完整的数据准备和实验命令见根目录 `README.md`。

时期内采样换成覆盖整个 30 秒的等间隔片段。时期间随机掩码、全局提示、Lion、余弦 warmup、100 epoch，以及损失 CE + 2×cosine + 0.5×transition CE，与 MASS 一致。Checkpoint 按准确率选取。

## 预算

三档预算与 MASS 表 1 使用同一组 \((r_a, r_e)\)。等间隔采样只改这 \(k\) 个 patch 的位置，不改数量，也不改时期间掩码。

| `--budget` | \(r_a\) | \(r_e\) | 可见 patch × epoch | 实际可见比例 |
|---|---|---|---|---|
| 0.72 | 0.2 | 0.1 | 24/30 × 28/32 | 0.70 |
| 0.40 | 0.5 | 0.2 | 15/30 × 25/32 | 0.390625 |
| 0.10 | 0.8 | 0.5 | 6/30 × 16/32 | 0.10 |

- 训练：随机相位，默认 25% 的 batch 使用 MASS 式随机时期内掩码（`--mix_random 0.25`）。
- 测试：相位固定为 0。
- `--pi random`：MASS 随机双层掩码，用作对照。
- `--pi uniform`：固定 linspace，无相位抖动。
- `--pi duty_cycle`：每个 30 秒只保留开头若干秒。
- `--pi band`：只保留指定 EEG 频带。
- `--no_align_inter`：32 个睡眠期全部可见。主实验不要加这个开关。

```bash
python -m SAS --smoke
python -m SAS --dataset dreams --pi strided --budget 0.40
python -m SAS --dataset dreams --pi random --budget 0.40
python -m SAS --dataset dreams --pi strided --budget 0.40 --fast --fold 0
```

结果写在 `SAS/results/<tag>_<时间>/`。
