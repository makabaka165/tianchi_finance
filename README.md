# 天池金融风控贷款违约预测解决方案

这是一个围绕天池金融风控贷款违约预测赛题搭建的完整解决方案仓库。

仓库里保留的不是单次提交代码，而是一整套可复现、可回看、可继续优化的本地实验体系，主要包括：

- LightGBM 基线与后续差异化实验
- CatBoost 主力方案
- 基于 OOF 的融合脚本
- 分阶段实验记录
- 高分方案总结文档

## 一、比赛任务

任务目标：根据借款人的贷款、信用、负债和历史行为特征，预测测试集样本的违约概率 `isDefault`。

评估指标：`AUC`

这个任务本质上是一个标准的风控表格二分类问题。模型最重要的不是把概率校准得非常精确，而是尽可能把高风险样本排在前面、低风险样本排在后面。

## 二、当前结果概览

### 1. 本地 OOF 结果

| 阶段 | 本地 OOF AUC | 说明 |
| --- | ---: | --- |
| LightGBM 初始基线 | `0.737236` | 实验 001 |
| LightGBM + 目标编码 | `0.739345` | 实验 003 |
| LightGBM 同系融合 | `0.741039` | 实验 009 |
| LightGBM + CatBoost 融合 | `0.7436926207152882` | 实验 012 |
| 按保留规则更新的正式最优 | `0.746763003490295` | `EXP023` |
| 本地已验证最高候选 | `0.7470961136601179` | `EXP034 none` |

### 2. 官网已提交结果

仓库中单独保留了官网提交留档，见 `official_submissions/README.md`。

当前已知最高官网分数对应文件：

- 文件：`official_submissions/submission_blend_exp034_pair_none.csv`
- 官网分数：`0.7468`

## 三、最终方案概述

当前最有效的主线不是不断增加模型数量，而是：

**LightGBM 打底 + CatBoost 主力建模 + 论坛启发的无泄漏特征工程 + 少量高价值类别组合 + 基于 OOF 的线性融合**

### 1. 按规则保留的正式最优方案

- 基础融合：`output/outputs_blend_exp012`
- 主力模型：`outputs_cat_exp023_forum`
- 融合权重：`0.09 / 0.91`
- 本地 OOF AUC：`0.746763003490295`

对应输出：

- `outputs_blend_exp023/submission_blend_exp023_lgb_cat_forum.csv`
- `outputs_blend_exp023/metrics_blend_exp023_lgb_cat_forum.json`

### 2. 本地已验证最高候选方案

- 基础：`outputs_blend_exp023`
- 增强模型：`outputs_cat_exp033_combo`
- 最优双模型融合权重：约 `0.495 / 0.505`
- 本地 OOF AUC：`0.7470961136601179`

对应输出：

- `outputs_blend_exp034_none/submission_blend_exp034_pair_none.csv`
- `outputs_blend_exp034_none/metrics_blend_exp034_pair_none.json`

## 四、真正有效的提分措施

## 1. 先建立稳定的 LightGBM 基线

`baseline_lgb.py` 提供了整个项目的基础训练骨架，主要包含：

- 5 折 `StratifiedKFold` OOF 验证
- 日期特征
- 等级映射特征
- 比例特征
- 计数特征
- 分组统计特征
- 可选的 CV 安全目标编码

这条线的主要价值有两个：

1. 建立稳定、可复现的表格模型基线
2. 为后续 CatBoost 融合提供差异来源

## 2. CatBoost 成为主力模型

后续最大的提升来自 `train_catboost.py` 对 CatBoost 路线的强化。

主要原因：

- 对高基数类别特征的处理更自然
- 对类别与数值交互的建模能力更强
- 在当前风控表格场景下，类别信息利用效率高于早期 LightGBM 基线

## 3. 论坛启发特征包是最关键升级

`EXP023` 的核心提升来自论坛方案中真正有价值、且不泄漏标签的信息，而不是直接照搬帖子。

关键特征包括：

### 分箱特征

- `employmentLength_bin`
- `issueDate_bin`
- `interestRate_bin`
- `annualIncome_bin`
- `loanAmnt_bin`
- `dti_bin`
- `installment_bin`
- `revolBal_bin`
- `revolUtil_bin`

### 比例特征

- `installment_term_revolBal`
- `revolUtil_revolBal`
- `openAcc_totalAcc`
- `loanAmnt_dti_annualIncome`
- `annualIncome_loanAmnt`

### 时间窗口统计

- `*_issueDate_median`
- `*_issueDate_ratio`

### 同群体相对位置特征

按以下群体生成分组中位数比例：

- `employmentLength`
- `purpose`
- `homeOwnership`

### 高 PSI 黑名单剔除

通过删除部分不稳定特征，降低漂移风险。

这些特征本质上都在强化几类关键信号：

- 偿债压力
- 额度利用率
- 收入与借款规模关系
- 信用历史成熟度
- 同时期、同群体下的相对异常程度

## 4. 少量高质量类别组合带来最后一段有效增益

`EXP033` 在 `EXP023` 基础上只增加了 4 组论坛类别组合：

- `grade__purpose`
- `subGrade__homeOwnership`
- `purpose__verificationStatus`
- `issueDate_bin__subGrade`

这一步非常克制，但收益明确：

- 单模型 AUC：`0.7467878225714177`
- 融合后 AUC：`0.7470960981385554`

说明少量高质量交互特征，比大规模特征扩张更有效。

## 5. 基于 OOF 的线性融合始终最好

后期依次验证过：

- 双模型线性融合
- 更细粒度网格搜索
- 排序变换融合
- `logit` 变换融合
- 三模型融合

结论很明确：

- 普通线性融合最好
- 复杂变换没有带来实质收益
- 额外模型只有在确实提供新信息时才有价值

## 五、被证明收益较低的方向

后期很多方向都跑通过，但收益有限，或者直接失败：

- LightGBM 差异化补充模型：能跑通，但与主力 CatBoost 的互补性不够强
- 论坛特征筛选：`EXP032` 虽然过了 smoke，但 full 明显变弱
- 基于对抗验证的漂移剔除：AUC 接近 `0.5`，说明不是当前主瓶颈
- 过度压缩类别组合：`EXP036` 反而掉分
- 轻量 combo-stat 增补：`EXP037` smoke 未过
- 排序融合与 `logit` 融合：都不如简单线性融合

这也是当前仓库最终停留在“CatBoost 主线 + 少量类别组合增强”的原因。

## 六、仓库结构

```text
.
├─ baseline_lgb.py                 # LightGBM 基线与变体
├─ train_catboost.py               # CatBoost 主力方案
├─ blend_predictions.py            # OOF / submission 融合脚本
├─ environment.yml                 # Conda 环境
├─ 实验记录.md                     # 完整实验记录
├─ 比赛总结与高分方案.md           # 详细复盘与方法总结
├─ PLAN.md                         # 目标模式实验规则
├─ official_submissions/           # 官网提交留档
├─ output/                         # 历史输出目录
└─ outputs_*/                      # 各轮实验产物
```

## 七、环境准备

推荐使用 Conda / Miniforge：

```powershell
conda env create -f environment.yml
conda activate tianchi_finance
```

如果本机没有将 `conda` 加入 PATH，可以直接使用完整路径：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe env create -f environment.yml
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py
```

## 八、数据文件

本仓库不提交原始比赛数据，请自行放在项目根目录：

- `train.csv`
- `testA.csv`
- `sample_submit.csv`

## 九、如何复现

## 1. 运行 LightGBM 基线

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py
```

快速冒烟测试：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py --sample-rows 50000 --n-splits 3 --n-estimators 300 --early-stopping-rounds 30 --output-dir outputs_smoke
```

## 2. 运行 `EXP023` 强力 CatBoost 方案

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python train_catboost.py --train-path E:\tianchi_finance\train.csv --test-path E:\tianchi_finance\testA.csv --iterations 2500 --learning-rate 0.05 --depth 6 --l2-leaf-reg 5 --early-stopping-rounds 180 --output-dir E:\tianchi_finance\outputs_cat_exp023_forum --run-name cat_exp023_forum --numeric-category-cols --forum-features
```

## 3. 复现正式最优融合

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python blend_predictions.py --oof E:\tianchi_finance\output\outputs_blend_exp012\oof_blend_exp012_lgb_cat_grid001.csv E:\tianchi_finance\outputs_cat_exp023_forum\oof_cat_exp023_forum.csv --sub E:\tianchi_finance\output\outputs_blend_exp012\submission_blend_exp012_lgb_cat_grid001.csv E:\tianchi_finance\outputs_cat_exp023_forum\submission_cat_exp023_forum.csv --weights 0.09 0.91 --output-dir E:\tianchi_finance\outputs_blend_exp023_rebuild --run-name blend_exp023_rebuild
```

## 4. 复现本地最高候选双模型融合

先跑出 `EXP033`：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python train_catboost.py --train-path E:\tianchi_finance\train.csv --test-path E:\tianchi_finance\testA.csv --iterations 2500 --learning-rate 0.05 --depth 6 --l2-leaf-reg 5 --early-stopping-rounds 180 --output-dir E:\tianchi_finance\outputs_cat_exp033_combo --run-name cat_exp033_combo --numeric-category-cols --forum-features --forum-category-combos v1
```

再做双模型融合：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python blend_predictions.py --oof E:\tianchi_finance\outputs_blend_exp023\oof_blend_exp023_lgb_cat_forum.csv E:\tianchi_finance\outputs_cat_exp033_combo\oof_cat_exp033_combo.csv --sub E:\tianchi_finance\outputs_blend_exp023\submission_blend_exp023_lgb_cat_forum.csv E:\tianchi_finance\outputs_cat_exp033_combo\submission_cat_exp033_combo.csv --search-step 0.005 --output-dir E:\tianchi_finance\outputs_blend_exp034_none_rebuild --run-name blend_exp034_pair_none
```

## 十、训练流程

后期实验采用固定流程：

1. 先在 `实验记录.md` 追加本轮计划
2. 先跑 smoke test
3. smoke 达标后才跑 full 5 折
4. full 完成后必须与当前 best 直接比较或融合
5. 验证 submission 格式
6. 只用本地 OOF AUC 判断是否保留

这个流程的好处是：

- 节省长时间 full 训练成本
- 降低无效试错
- 每一步都有明确记录与回退依据

## 十一、建议阅读顺序

如果你只想快速理解这套方案，建议按下面顺序阅读：

1. `比赛总结与高分方案.md`
2. `实验记录.md`
3. `train_catboost.py`
4. `baseline_lgb.py`
5. `blend_predictions.py`

## 十二、说明

- 仓库中的 `outputs_*` 目录主要用于保留实验产物和复盘，不建议全部提交到 Git。
- 最终高分方案更依赖特征表达质量和模型差异，而不是更复杂的融合框架。
- 如果后续继续提分，优先方向仍应围绕当前 CatBoost 主线做高质量小改动，而不是推翻整个流程。

## 十三、使用建议

这份仓库更适合作为比赛复盘和方法借鉴材料使用。

如果你准备基于这套代码继续训练，建议优先保持以下原则：

- 保留 OOF 驱动的验证与融合方式
- 控制新增特征的质量而不是数量
- 尽量避免引入目标泄漏风险
- 所有提升都用本地 OOF AUC 说话
