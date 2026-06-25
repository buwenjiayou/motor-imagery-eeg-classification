# SHU 运动想象 EEG 分类研究

本项目是上海大学课程设计，围绕 SHU 运动想象 EEG 数据集，系统开展脑电信号的预处理、可视化分析、传统空间滤波方法复现与改进，以及深度学习数据准备工作。核心研究难点是跨会话（cross-session）分布漂移：同一被试在不同采集日期的 EEG 分布存在明显差异，导致 session 内表现良好的分类器在跨 session 测试时性能急剧下降。

## 项目简介

运动想象（Motor Imagery, MI）脑电信号的分类是脑机接口（BCI）领域的经典问题。本项目以 CSP（Common Spatial Pattern）、FBCSP（Filter Bank CSP）等传统方法为基线，在严格无泄漏协议下评估 within-session 与 cross-session 性能差异，并尝试通过稳定性加权频带融合（Stable-FBCSP）缓解跨 session 漂移。

项目的科研训练价值体现在：
- 建立可信基线：所有评估遵守无泄漏规则，测试 session 仅用于最终评估；
- 方法迭代反思：从特征尺度加权修正为频带级概率融合，发现 V1 稳定性估计不够严格后推进到 V2；
- 可视化与分类分离：信号分析中的 trial 清洗不影响分类实验的数据完整性；
- 完整实验链路：从数据审计、信号可视化、传统方法、方法改进到深度学习数据准备，覆盖科研项目的完整生命周期。

## 数据集

本项目使用 **SHU motor imagery dataset**，包含 25 名被试，每名被试完成 5 个采集会话（session）。数据以三种形式存放于 `data/raw/` 目录下：

| 目录 | 格式 | 说明 |
|---|---|---|
| `data/raw/mat/` | `.mat` | 正式分类实验使用的 trial 级数据，每文件含 `data`（trials × channels × time）和 `labels` |
| `data/raw/edf/` | `.edf` | 通用 EEG 连续数据格式，与 MAT 对应同一批 session |
| `data/raw/events/` | `.tsv` | EDF 对应的事件标注文件，采用 BIDS-like 格式 |

**三者关系**：`mat/`、`edf/` 和 `events/` 不是三个独立数据集，而是同一数据集的不同表示或辅助标注。当前分类实验主要读取 `.mat` 文件。

**数据审计结果**：

| 项目 | 数量 |
|---|---:|
| MAT 文件 | 125 |
| EDF 文件 | 125 |
| events 文件 | 125 |
| 被试数 | 25 |
| session 总数 | 125 |
| 通道数 | 32 |
| events trial 总数 | 11,988 |
| 左手想象 trial | 5,983 |
| 右手想象 trial | 6,005 |
| 不完整 session 记录 | 0 |

> 注：审计过程中发现 `sub-001_ses-04_task_motorimagery_events.tsv` 缺少表头，但事件内容有效，审计脚本已兼容该情况。

## 目录结构

```text
motor-imagery-eeg-classification/
├── code/                          # 老师提供的原始代码
│   ├── cs/                        # Cross-subject FBCNet
│   ├── csa/                       # Cross-session adaptation
│   ├── ws/                        # Within-session CSP/FBCSP/FBCNet
│   ├── preprocess/                # 预处理脚本
│   ├── ERD_ERS.py                 # ERD/ERS 原始脚本
│   └── ...
├── data/
│   ├── raw/                       # 原始数据（只读）
│   │   ├── mat/                   # .mat trial 级数据
│   │   ├── edf/                   # EDF 连续数据
│   │   ├── events/                # 事件标注
│   │   └── metadata/              # 通道、被试等元数据
│   ├── interim/                   # 中间处理结果
│   └── processed/                 # 可直接用于实验的数据
│       └── deep_learning/         # 深度学习 NPZ 导出
├── experiments/                   # 实验脚本（E0-E18，部分序号预留/合并）
├── results/
│   ├── figures/                   # 图表输出
│   ├── tables/                    # CSV 结果表格
│   ├── logs/                      # JSON 实验日志
│   └── dl/                        # 深度学习训练日志
├── docs/
│   └── report/                    # 课程报告与实验记录
├── scripts/                       # 辅助脚本（如数据链接修复）
├── .gitignore
└── README.md
```

**数据约定**：
- `data/raw/` 中的文件视为只读，不在原文件上修改或覆盖；
- 滤波、分段或格式转换的临时结果写入 `data/interim/`；
- 最终实验输入写入 `data/processed/`；
- 训练结果、图片和模型统一写入 `results/`。

**兼容原始代码**：原始脚本默认从 `code/SHU_Dataset/` 读取 `.mat` 文件。当前该路径是一个 Windows 目录联接，实际指向 `data/raw/mat/`。如果联接失效，在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_data_link.ps1
```

## 环境配置

推荐使用 Conda 创建独立环境：

```powershell
conda create -n mi-eeg-cpu python=3.11
conda activate mi-eeg-cpu
```

关键依赖：

| 包 | 用途 |
|---|---|
| numpy | 数值计算 |
| scipy | 信号处理、MAT 文件读取 |
| scikit-learn | CSP、LDA、交叉验证、指标计算 |
| matplotlib | 可视化 |
| tqdm | 进度条 |

安装命令示例：

```powershell
conda activate mi-eeg-cpu
pip install numpy scipy scikit-learn matplotlib tqdm
```

> 当前 CPU 实验环境不安装 PyTorch；深度学习实验（EEGNet/FBCNet）在单独的 GPU 环境中进行。

## 实验复现

所有实验脚本位于 `experiments/` 目录，按顺序执行即可复现完整结果。

### E0 数据审计

验证 MAT/EDF/events 文件覆盖率和事件标签分布。

```powershell
conda activate mi-eeg-cpu
python experiments/00_data_audit.py
```

输出：`results/tables/data_audit_summary.csv`、`results/tables/data_audit_by_session.csv`

### E1 信号可视化（ERD/ERS、PSD、频带功率）

生成 C3/C4 通道的 ERD/ERS-style 相对功率动态、Welch PSD 频谱曲线和频带功率统计图。

```powershell
conda activate mi-eeg-cpu
python experiments/08_signal_visualization.py --all-subjects --mode full --artifact-threshold 100 --artifact-scope target
```

输出：
- `results/figures/erd_ers_style_c3_c4_*.png`
- `results/figures/psd_c3_c4_*.png`
- `results/figures/band_power_c3_c4_*.png`
- `results/tables/signal_band_power_*.csv`
- `results/tables/signal_relative_power_*.csv`

> 注意：`--artifact-threshold 100` 仅在可视化阶段剔除 C3/C4 上最大绝对振幅超过 100 μV 的 trial，不影响分类实验。

### E2/E3 CSP-LDA / FBCSP-LDA 基线

复现传统基线，包含 within-session 5-fold 交叉验证和 cross-session leave-one-session-out（训练 1-4，测试 5）。

```powershell
conda activate mi-eeg-cpu
python experiments/02_cpu_csp_fbcsp_baseline.py --all-subjects --mode full --folds 5 --seed 2026
```

汇总结果：

```powershell
python experiments/03_summarize_results.py `
  --input results/tables/cpu_csp_fbcsp_full_all_subjects.csv `
  --output results/tables/cpu_csp_fbcsp_full_summary.csv
```

输出：
- `results/tables/cpu_csp_fbcsp_full_all_subjects.csv`
- `results/tables/cpu_csp_fbcsp_full_summary.csv`

### E4 Stable-FBCSP V1（频带级概率融合，λ=0.5/1.0）

基于训练 session 内半分验证估计频带判别性与稳定性，使用频带级概率融合进行 cross-session 测试。

```powershell
conda activate mi-eeg-cpu
python experiments/04_stable_fbcsp.py --all-subjects --mode full --lambda-stability 0.5
python experiments/04_stable_fbcsp.py --all-subjects --mode full --lambda-stability 1.0
```

汇总结果（以 λ=0.5 为例）：

```powershell
python experiments/03_summarize_results.py `
  --input results/tables/stable_fbcsp_full_all_subjects_lambda_0p5.csv `
  --output results/tables/stable_fbcsp_full_summary_lambda_0p5.csv
```

输出：
- `results/tables/stable_fbcsp_full_all_subjects_lambda_0p5.csv`
- `results/tables/stable_fbcsp_full_summary_lambda_0p5.csv`
- `results/tables/stable_fbcsp_weights_full_all_subjects_lambda_0p5.csv`

### E5 Stable-FBCSP V2（inner leave-one-session-out 频带权重）

在训练 session 1-4 内部做 leave-one-session-out 验证，估计每个频带的跨 session 泛化能力。

```powershell
conda activate mi-eeg-cpu
python experiments/07_stable_fbcsp_v2.py --all-subjects --mode full --lambda-stability 0.5
```

汇总结果：

```powershell
python experiments/03_summarize_results.py `
  --input results/tables/stable_fbcsp_v2_full_all_subjects_lambda_0p5.csv `
  --output results/tables/stable_fbcsp_v2_full_summary_lambda_0p5.csv
```

输出：
- `results/tables/stable_fbcsp_v2_full_all_subjects_lambda_0p5.csv`
- `results/tables/stable_fbcsp_v2_full_summary_lambda_0p5.csv`
- `results/tables/stable_fbcsp_v2_weights_full_all_subjects_lambda_0p5.csv`

### E6 图表生成

从已完成的实验表格生成报告级对比图。

```powershell
conda activate mi-eeg-cpu
python experiments/05_merge_cross_session_results.py
python experiments/06_make_report_figures.py
```

输出：
- `results/tables/cross_session_method_comparison.csv`
- `results/figures/cross_session_method_comparison.png`
- `results/figures/within_vs_cross_session.png`
- `results/figures/subject_cross_session_distribution.png`
- `results/figures/stable_fbcsp_*_lambda_0p5.png`
- `results/figures/stable_fbcsp_v2_*_lambda_0p5.png`

### E7 深度学习数据准备（NPZ 导出）

将 MAT 数据导出为 NumPy NPZ 格式，供后续 EEGNet/FBCNet 实验使用。

```powershell
conda activate mi-eeg-cpu
python experiments/09_prepare_deep_learning_data.py --all-subjects --output shu_mi_25subjects_5sessions.npz
```

输出：
- `data/processed/deep_learning/shu_mi_25subjects_5sessions.npz`
- `results/logs/prepare_deep_learning_data.json`

## 主要结果

### Within-session vs Cross-session 基线

| 协议 | 方法 | n | Balanced Accuracy |
|---|---|---:|---:|
| within-session 5-fold | CSP-LDA | 125 | 0.6034 ± 0.1617 |
| within-session 5-fold | FBCSP-LDA | 125 | 0.6056 ± 0.1331 |
| cross-session 1-4→5 | CSP-LDA | 25 | 0.5303 ± 0.0652 |
| cross-session 1-4→5 | FBCSP-LDA | 25 | 0.5061 ± 0.0364 |

### Cross-session 方法对比

| 方法 | λ | Balanced Accuracy |
|---|---:|---:|
| CSP-LDA | - | **0.5303 ± 0.0652** |
| FBCSP-LDA | - | 0.5061 ± 0.0364 |
| FBCSP-equal | 0.5 | 0.5152 ± 0.0504 |
| FBCSP-discriminative-weighted | 0.5 | 0.5154 ± 0.0543 |
| Stable-FBCSP | 1.0 | 0.5042 ± 0.0590 |
| Stable-FBCSP | 0.5 | 0.5115 ± 0.0638 |
| FBCSP-innerLOSO-weighted | 0.5 | 0.5132 ± 0.0493 |
| Stable-FBCSP-V2 | 0.5 | 0.5181 ± 0.0613 |

关键观察：
- within-session 下 CSP-LDA 与 FBCSP-LDA 均达到约 0.60 的 balanced accuracy，说明数据中存在可学习的运动想象判别信息；
- cross-session 下性能明显下降到接近随机水平，session drift 是核心难点；
- 原始拼接式 FBCSP-LDA 在 cross-session 下不如 CSP-LDA；
- 频带级概率融合相比原始拼接式 FBCSP 有一定改善；
- Stable-FBCSP-V2（0.5181）相比 V1（0.5115）有轻微提升，但仍未超过 CSP-LDA 基线。

## 核心发现与局限性

1. **CSP-LDA 仍是 cross-session 最强传统基线**
   在 25 名被试的 cross-session 协议下，CSP-LDA 的 balanced accuracy（0.5303）高于所有 FBCSP 变体。这说明对于当前数据，简单的全频段 CSP 空间滤波在跨 session 泛化上反而比多频带扩展更稳健。

2. **Stable-FBCSP 有轻微改善趋势但不显著**
   V2 相比 V1 从 0.5115 提升到 0.5181，相比 FBCSP-equal 平均提升约 +0.0028，但 bootstrap 95% CI 跨过 0，不能声称显著提升。该结果应定位为"基于训练内跨 session 验证的频带稳定性加权尝试，存在轻微改善趋势"。

3. **跨会话稳定性估计需要更严格的 inner LOSO**
   V1 使用单个训练 session 内部半分验证的波动来估计稳定性，这与最终 cross-session 任务不完全一致。V2 改用训练 session 内部的 leave-one-session-out，更贴近目标任务，因此表现更合理。

4. **可视化清洗与分类实验严格分离**
   信号可视化中的 C3/C4 振幅阈值剔除（100 μV）仅在 E1 中使用，不影响 E2-E7 的分类实验数据。这保证了信号探索的灵活性与分类评估的严谨性互不干扰。

## 后续工作

若以优秀课程设计为目标，当前还需要完成：
- 报告方法章节与实验章节整理；
- 将本阶段图表放入报告；
- 准备答辩 PPT。

若以开源科研训练项目为目标，建议下一步优先完成：

1. **EA 对齐**：尝试 Euclidean Alignment 缓解 session drift，已在 `experiments/15_ea_csp_baseline.py`、`16_ea_fbcsp_baseline.py`、`17_ea_eegnet_compressed.py` 中初步探索；
2. **深度学习扩展**：EEGNet、FBCNet 的 cross-session 实验，已在 `experiments/10_eegnet_smoke.py` 中完成 smoke test；
3. **代码重构**：将可复用代码从 `experiments/` 逐步整理到 `src/mi_eeg/`；
4. **开源整理**：完善 LICENSE、CITATION、复现实验说明。

更合理的路线是：先把传统方法、评估协议和跨 session 问题讲清楚，再用深度学习作为扩展对照。这样项目会更像一个完整科研训练项目，而不是单纯"跑了几个模型"。

## 开源策略

本项目计划将代码和文档开源到 GitHub，但原始 EEG 数据体积过大，不随仓库分发。

**提交到 GitHub 的内容**：
- `experiments/`：所有实验脚本
- `docs/`：课程报告与实验记录
- `scripts/`：辅助脚本
- `results/figures/` 和 `results/tables/`：关键图表和汇总表格
- `README.md`、`.gitignore`

**通过 `.gitignore` 排除的内容**：
- `data/raw/mat/`、`data/raw/edf/`、`data/raw/events/`：原始数据（体积大，需单独获取）
- `data/interim/`、`data/processed/`：中间和最终处理数据
- `results/logs/*.json`：详细日志（可选排除，保留结构）
- `*.pt`、`*.pth`、`*.h5`、`*.pkl`：模型权重文件
- IDE 和 OS 临时文件

获取原始数据后，在项目根目录运行 `scripts/setup_data_link.ps1` 即可恢复原始代码的兼容路径。

## 引用与许可

本项目为上海大学课程设计，基于老师提供的 SHU 运动想象 EEG 数据集和原始算法代码（CSP、FBCSP、FBCNet、EEGNet 等）进行复现与扩展。

如果你在本项目代码基础上开展研究，建议引用以下相关文献：
- Ramoser et al. (2000). "Optimal spatial filtering of single trial EEG during imagined hand movement." *IEEE TBME*.
- Ang et al. (2008). "Filter bank common spatial pattern (FBCSP) in brain-computer interface." *IJCNN*.
- Lawhern et al. (2018). "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces." *JNE*.

原始数据集和基础代码的版权归属原作者/原单位。本项目新增实验脚本和文档遵循 MIT 许可证（如单独声明）。
