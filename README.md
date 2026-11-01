# SAS-MASS

基于单通道脑电等间隔采样的可穿戴睡眠分期（Wearable sleep staging based on strided sampling of single-channel EEG）。

本仓库是该方法的训练与评估代码。分期网络沿用掩码感知睡眠分期（Mask-Aware Sleep Staging, MASS）：时期间随机掩码，以及由可见片段汇聚而成的全局提示。时期内不再随机抽取 1 秒片段，而改为覆盖整个 30 秒的等间隔采样。训练时在一个步长内随机平移相位，并让一部分 batch 使用 MASS 式随机时期内掩码；测试与部署时相位固定为 0，采集和待机时刻可以事先写入定时器。

论文实验请用 `python -m SAS` 复现。根目录的 `main_*.py` 是上游 MASS 的原始入口，保留用于对照，不产生本文的等间隔采样结果。

## 代码来源

网络骨架、数据读取和 MASS 随机掩码训练来自：

- Lejun Ai et al. *Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning.* AAAI 2026.
- 代码：<https://github.com/AnsonAiTRAY/MASS>
- 扩展版本：<https://arxiv.org/abs/2511.06785>

原仓库为 MIT 许可，版权声明见本仓库 `LICENSE`。本文的改动在 `SAS/`：等间隔采样策略、相位抖动，以及与 MASS 表 1 相同预算下的训练入口。

## 目录

| 路径 | 内容 |
|---|---|
| `SAS/` | 本文方法。`python -m SAS` |
| `structure/mass.py` | MASS 分期网络。`SAS` 在其上替换时期内采样 |
| `structure/mask.py`, `structure/transformer_layers.py` | MASS 掩码与 Transformer |
| `utils.py` | `.mat` 读取与 DataLoader |
| `train.py`, `main_*.py` | 上游 MASS 训练入口 |
| `preprocess/` | 原始 EDF 转为训练用 `.mat`，含 SHHS 的 329 人名单 |
| `others/LGSleepNet/model/Polyloss.py` | 仅供 `train.py` 导入。本文损失不用 PolyLoss |
| `dataset/` | 本地数据目录，不纳入版本库 |

`structure/mae_*.py`、`structure/patch.py`、`structure/stft.py` 是随 MASS 骨架保留的上游文件，本文训练不调用它们。

## 环境

建议 Python 3.10+。先安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖：

```bash
pip install torch
pip install -r requirements.txt
```

Windows 上若出现 `fbgemm.dll` / WinError 127，使用已经能运行 MASS 的环境。`SAS/_env.py` 会在导入 NumPy 之前先导入 PyTorch。

无数据通路检查：

```bash
python -m SAS --smoke
```

## 数据

原始 EDF 和中间 `.mat` 都不要提交到 GitHub。把预处理后的文件放在仓库根目录：

```text
dataset/DREAMS/subject*.mat          # 键: PSD, label
dataset/EDF20/SC4*.mat               # 键: PSD100, label
dataset/EDF78/SC4*.mat               # 键: PSD100, label
dataset/SHHS/shhs1-*.mat             # 键: PSD, label
```

预处理在仓库根目录执行。脚本会把结果写回上述目录。

```bash
python preprocess/preprocess.py --dataset dreams --input_dir <DREAMS EDF 目录>
python preprocess/preprocess.py --dataset edf20 --input_dir <Sleep-EDF cassette 目录>
python preprocess/preprocess.py --dataset edf78 --input_dir <Sleep-EDF cassette 目录>
python preprocess/preprocess.py --dataset shhs --input_dir <shhs1 EDF 目录> --ann_dir <shhs1 Profusion XML 目录>
```

SHHS 默认使用 `preprocess/selected_shhs1_files.txt`（329 条 `shhs1-*` 记录，与 MASS / AttnSleep 子集一致）：C4-A1，125 Hz 重采样到 100 Hz，N3 与 N4 合并，去掉未知分期，睡眠起止各保留 30 分钟清醒。可用 `--limit 1` 先处理一条做检查。

数据需自行申请，本仓库不提供下载：

- DREAMS Subjects Database
- Sleep-EDF Expanded：<https://physionet.org/content/sleep-edfx/>
- SHHS：<https://sleepdata.org/datasets/shhs>

划分与 MASS 一致。DREAMS 为 20 名受试者留一法。Sleep-EDF-20、Sleep-EDF-78 和 SHHS 为 20 折受试者划分。默认 batch size 分别为 8、16、64、128。

## 复现本文实验

在仓库根目录运行。三档预算与 MASS 表 1 使用同一组 \((r_a, r_e)\)，`ceil` 之后的可见片段数也相同。等间隔采样只改变这 \(k\) 个片段在 30 秒内的位置。

| `--budget` | \(r_a\) | \(r_e\) | 可见 patch × epoch | 实际可见比例 |
|---|---|---|---|---|
| 0.72 | 0.2 | 0.1 | 24/30 × 28/32 | 0.70 |
| 0.40 | 0.5 | 0.2 | 15/30 × 25/32 | 0.390625 |
| 0.10 | 0.8 | 0.5 | 6/30 × 16/32 | 0.10 |

训练配方与 MASS 相同：Lion、余弦 warmup、100 epoch，损失为交叉熵 + 2×cosine + 0.5×transition 交叉熵，按准确率保存 checkpoint。`strided` 默认 `--mix_random 0.25`。

```bash
# DREAMS，等间隔采样
python -m SAS --dataset dreams --pi strided --budget 0.72
python -m SAS --dataset dreams --pi strided --budget 0.40
python -m SAS --dataset dreams --pi strided --budget 0.10

# Sleep-EDF-20 / Sleep-EDF-78 / SHHS
python -m SAS --dataset edf20 --pi strided --budget 0.40
python -m SAS --dataset edf78 --pi strided --budget 0.40
python -m SAS --dataset shhs --pi strided --budget 0.40

# 同一预算下的 MASS 随机掩码对照
python -m SAS --dataset dreams --pi random --budget 0.40

# 只跑第 0 折、2 个 epoch
python -m SAS --dataset dreams --pi strided --budget 0.40 --fast --fold 0
```

四个数据集都把 `--budget` 换成 `0.72`、`0.40`、`0.10`。随机对照把 `--pi strided` 换成 `--pi random`。

其它 `--pi`：`uniform` 为固定 linspace、无相位抖动；`duty_cycle` 为每个 30 秒只保留开头若干秒；`band` 为只保留指定频带。`--no_align_inter` 会让 32 个睡眠期全部可见。本文主实验使用默认的 `strided`，并保留时期间掩码。

结果写在 `SAS/results/<dataset>_<pi>_B0.40_<时间>/`，包含 `summary.json`、`fold*.json`、checkpoint 和预测。汇总指标为 ACC、MF1、κ、MGm。该目录已在 `.gitignore` 中排除。

直接运行上游入口时，先建日志目录，否则 `FileHandler` 会失败：

```bash
mkdir log board model
python main_dreams.py --intra_mask_ratio 0.5 --inter_mask_ratio 0.2
python main_edf20.py --intra_mask_ratio 0.5 --inter_mask_ratio 0.2
python main_edf78.py --intra_mask_ratio 0.5 --inter_mask_ratio 0.2
python main_shhs.py --intra_mask_ratio 0.5 --inter_mask_ratio 0.2
```

## 许可与引用

本仓库以 MIT 许可发布，见 `LICENSE`。其中保留原 MASS 版权（AnsonAiTRAY，2025）和本仓库版权（WGeorge24，2026）。分发时请保留这两条声明。

使用本代码请引用本文，并同时引用 MASS（Ai et al., AAAI 2026）。本文正式出版信息确定后可补到此处。
