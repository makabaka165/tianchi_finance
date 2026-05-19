# Tianchi Finance Default Prediction Solution

天池金融风控贷款违约预测解决方案仓库。

这个仓库不是单一脚本提交版，而是一套完整的本地实验与提分流程，包含：

- LightGBM 基线与差异化实验
- CatBoost 主力方案
- OOF 融合脚本
- 分阶段实验记录
- 最终高分方案总结

## 1. Competition

任务目标：基于借款人贷款、信用、负债和历史行为特征，预测测试集样本的违约概率 `isDefault`。

评估指标：`AUC`

这个任务本质上是一个典型的表格风控二分类问题，重点不是概率绝对值校准，而是风险排序能力。

## 2. Results Overview

### Local OOF

| Type | AUC | Notes |
| --- | ---: | --- |
| Early LightGBM baseline | `0.737236` | 实验 001 |
| LightGBM + target encoding | `0.739345` | 实验 003 |
| LGB same-family blend | `0.741039` | 实验 009 |
| LGB + CatBoost blend | `0.7436926207152882` | 实验 012 |
| Official best under keep rules | `0.746763003490295` | `EXP023` |
| Best validated local candidate | `0.7470961136601179` | `EXP034 none` |

### Official submissions kept in this repo

见 `official_submissions/README.md`。

当前已知最高官网提交分数：

- file: `official_submissions/submission_blend_exp034_pair_none.csv`
- public score: `0.7468`

## 3. Final Solution Summary

当前整套方案里，真正有效的主线不是继续堆更多模型，而是：

**LightGBM 打底 + CatBoost 主力建模 + 论坛启发的无泄漏特征工程 + 少量高价值类别组合 + 基于 OOF 的线性融合**

对应高分方案可以拆成两层：

### 3.1 Official best

- base blend: `output/outputs_blend_exp012`
- strong model: `outputs_cat_exp023_forum`
- blend weight: `0.09 / 0.91`
- local OOF AUC: `0.746763003490295`

输出文件：

- `outputs_blend_exp023/submission_blend_exp023_lgb_cat_forum.csv`
- `outputs_blend_exp023/metrics_blend_exp023_lgb_cat_forum.json`

### 3.2 Best validated local candidate

- base: `outputs_blend_exp023`
- enhanced CatBoost: `outputs_cat_exp033_combo`
- best pair blend: about `0.495 / 0.505`
- local OOF AUC: `0.7470961136601179`

输出文件：

- `outputs_blend_exp034_none/submission_blend_exp034_pair_none.csv`
- `outputs_blend_exp034_none/metrics_blend_exp034_pair_none.json`

## 4. What Actually Worked

### 4.1 Stable LightGBM baseline

`baseline_lgb.py` 提供了整个项目的基础骨架：

- `StratifiedKFold` 5 折 OOF 验证
- 日期特征
- 等级映射特征
- 比例特征
- 计数特征
- group stat 特征
- 可选 CV-safe target encoding

这条线的主要作用是：

1. 建立可复现 baseline
2. 形成和 CatBoost 融合的差异来源

### 4.2 CatBoost became the main model

后续最大的提升来自 `train_catboost.py`：

- 对高基数类别的处理更自然
- 类别与数值交互建模更强
- 在当前风控表格数据上，类别信息利用效率明显高于 LightGBM baseline

### 4.3 Forum-inspired feature pack was the key upgrade

`EXP023` 的高分核心来自论坛方案中的无监督、无泄漏特征，而不是直接照搬帖子。

关键特征包括：

- 分箱特征
  - `employmentLength_bin`
  - `issueDate_bin`
  - `interestRate_bin`
  - `annualIncome_bin`
  - `loanAmnt_bin`
  - `dti_bin`
  - `installment_bin`
  - `revolBal_bin`
  - `revolUtil_bin`
- 比例特征
  - `installment_term_revolBal`
  - `revolUtil_revolBal`
  - `openAcc_totalAcc`
  - `loanAmnt_dti_annualIncome`
  - `annualIncome_loanAmnt`
- 时间窗口统计
  - `*_issueDate_median`
  - `*_issueDate_ratio`
- 同群体相对位置
  - 按 `employmentLength`
  - 按 `purpose`
  - 按 `homeOwnership`
  生成分组中位数比例
- 高 PSI 黑名单剔除

这些特征本质上都在表达几件事：

- 偿债压力
- 额度利用率
- 收入与借款规模关系
- 信用历史成熟度
- 同时期、同群体下的相对异常程度

### 4.4 Small manual category combos added the last useful gain

`EXP033` 在 `EXP023` 基础上只增加了 4 个论坛类别组合：

- `grade__purpose`
- `subGrade__homeOwnership`
- `purpose__verificationStatus`
- `issueDate_bin__subGrade`

这一改动很克制，但有效：

- single model AUC: `0.7467878225714177`
- blend AUC: `0.7470960981385554`

说明少量高质量交互，比大规模特征扩张更值得做。

### 4.5 OOF-based linear blend remained the best blending strategy

后期验证过：

- pairwise linear blend
- finer grid search
- rank transform blend
- logit transform blend
- three-model blend

结论是：

- **普通线性融合最好**
- 复杂变换没有带来实质收益
- 额外模型只有在真有差异时才有价值

## 5. What Did Not Work Well

后期很多方向都跑通了，但收益非常有限，或者直接失败：

- LightGBM 差异化补充模型：能跑通，但和主力 CatBoost 的互补性不够强
- 论坛特征筛选：`EXP032` smoke 过线，但 full 明显变弱
- 对抗验证驱动的 drift-drop：AUC 接近 `0.5`，说明不是当前主瓶颈
- 过度压缩类别组合：`EXP036` 反而掉分
- 轻量 combo-stat 增补：`EXP037` smoke 未过
- `rank/logit` 融合变换：不如简单线性融合

这也是当前仓库最终停留在“CatBoost 主线 + 小幅组合增强”的原因。

## 6. Repository Structure

```text
.
├─ baseline_lgb.py                 # LightGBM baseline and variants
├─ train_catboost.py               # CatBoost main solution
├─ blend_predictions.py            # OOF/submission blending
├─ environment.yml                 # Conda environment
├─ 实验记录.md                     # Full experiment log
├─ 比赛总结与高分方案.md           # Detailed summary and methodology notes
├─ PLAN.md                         # Goal-mode experiment rules
├─ official_submissions/           # Archived website submissions
├─ output/                         # Historical outputs
└─ outputs_*/                      # Experiment artifacts
```

## 7. Environment

推荐使用 Conda / Miniforge：

```powershell
conda env create -f environment.yml
conda activate tianchi_finance
```

如果本机未将 `conda` 加入 PATH，可以直接使用完整路径：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe env create -f environment.yml
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py
```

## 8. Data

本仓库不提交原始比赛数据，请自行放在项目根目录：

- `train.csv`
- `testA.csv`
- `sample_submit.csv`

## 9. How To Reproduce

### 9.1 Run the LightGBM baseline

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py
```

Smoke test:

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py --sample-rows 50000 --n-splits 3 --n-estimators 300 --early-stopping-rounds 30 --output-dir outputs_smoke
```

### 9.2 Run the strong CatBoost forum model

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python train_catboost.py --train-path E:\tianchi_finance\train.csv --test-path E:\tianchi_finance\testA.csv --iterations 2500 --learning-rate 0.05 --depth 6 --l2-leaf-reg 5 --early-stopping-rounds 180 --output-dir E:\tianchi_finance\outputs_cat_exp023_forum --run-name cat_exp023_forum --numeric-category-cols --forum-features
```

### 9.3 Reproduce the official best blend

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python blend_predictions.py --oof E:\tianchi_finance\output\outputs_blend_exp012\oof_blend_exp012_lgb_cat_grid001.csv E:\tianchi_finance\outputs_cat_exp023_forum\oof_cat_exp023_forum.csv --sub E:\tianchi_finance\output\outputs_blend_exp012\submission_blend_exp012_lgb_cat_grid001.csv E:\tianchi_finance\outputs_cat_exp023_forum\submission_cat_exp023_forum.csv --weights 0.09 0.91 --output-dir E:\tianchi_finance\outputs_blend_exp023_rebuild --run-name blend_exp023_rebuild
```

### 9.4 Reproduce the strongest validated local pair

先跑出 `EXP033`：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python train_catboost.py --train-path E:\tianchi_finance\train.csv --test-path E:\tianchi_finance\testA.csv --iterations 2500 --learning-rate 0.05 --depth 6 --l2-leaf-reg 5 --early-stopping-rounds 180 --output-dir E:\tianchi_finance\outputs_cat_exp033_combo --run-name cat_exp033_combo --numeric-category-cols --forum-features --forum-category-combos v1
```

再做 pairwise blend：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python blend_predictions.py --oof E:\tianchi_finance\outputs_blend_exp023\oof_blend_exp023_lgb_cat_forum.csv E:\tianchi_finance\outputs_cat_exp033_combo\oof_cat_exp033_combo.csv --sub E:\tianchi_finance\outputs_blend_exp023\submission_blend_exp023_lgb_cat_forum.csv E:\tianchi_finance\outputs_cat_exp033_combo\submission_cat_exp033_combo.csv --search-step 0.005 --output-dir E:\tianchi_finance\outputs_blend_exp034_none_rebuild --run-name blend_exp034_pair_none
```

## 10. Training Workflow

后期实验采用固定流程：

1. 先在 `实验记录.md` 追加本轮计划
2. 先跑 smoke test
3. smoke 达标才跑 full 5-fold
4. full 完成后必须与当前 best 直接比较或融合
5. 验证 submission 格式
6. 只用本地 OOF AUC 判断是否保留

这个流程的好处是：

- 节省长时间 full 训练成本
- 降低无效试错
- 保证每一步都有记录和回退依据

## 11. Important Files

如果你只想快速理解这套方案，建议按下面顺序阅读：

1. `比赛总结与高分方案.md`
2. `实验记录.md`
3. `train_catboost.py`
4. `baseline_lgb.py`
5. `blend_predictions.py`

## 12. Notes

- 本仓库里的 `outputs_*` 目录主要用于保留实验产物和复盘，不建议全部提交到 Git。
- 最终高分方案更依赖特征表达质量和模型差异，而不是更复杂的融合框架。
- 如果后续继续提分，优先方向仍应围绕当前 CatBoost 主线做高质量小改动，而不是重新推翻整个流程。

## 13. License / Usage

本仓库更适合作为比赛复盘和方法借鉴材料使用。

如果你准备基于这套代码继续训练，建议优先保持以下原则：

- 保留 OOF 驱动的验证与融合方式
- 控制新增特征的质量而不是数量
- 尽量避免引入目标泄漏风险
- 所有提升都用本地 OOF AUC 说话
