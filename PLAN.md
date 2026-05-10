# PLAN: Goal Mode Execution Plan for AUC 0.75+

本文件用于 `/goal` 执行阶段。执行者必须按本计划工作，不要扩展目标，不要改动训练数据，不要消耗官网提交次数。

## 1. Goal

硬目标：

```text
local OOF AUC >= 0.75
```

阶段性保留目标：

```text
candidate_auc >= current_best_auc + 0.0005
```

当前最佳基准：

```text
current_best_auc: 0.7436926207152882
best_submission: outputs_blend_exp012/submission_blend_exp012_lgb_cat_grid001.csv
best_metrics: outputs_blend_exp012/metrics_blend_exp012_lgb_cat_grid001.json
best_commit: 4a2ceb9
best_method: 0.27 * outputs_blend_exp009 + 0.73 * outputs_cat_exp012
```

评价指标只使用本地 OOF AUC。不要以官网分数、训练集分数、单折分数或主观判断作为保留依据。

## 2. Environment

固定使用 Windows PowerShell 和已有 Miniforge 环境：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python -c "import pandas, numpy, sklearn, lightgbm, catboost; print('env ok')"
```

不要新建环境。除非进入 XGBoost 阶段，否则不要安装新依赖。

## 3. Scope

允许修改：

```text
baseline_lgb.py
train_catboost.py
blend_predictions.py
environment.yml
实验记录.md
PLAN.md
README.md
```

禁止修改：

```text
train.csv
testA.csv
sample_submit.csv
赛题.md
outputs*
```

不要提交 `outputs*`、CSV 数据、模型缓存、`__pycache__`。

## 4. Git Rules

开始执行前创建独立分支：

```powershell
git checkout -b goal-auc-075
```

如果分支已存在，切换到该分支：

```powershell
git checkout goal-auc-075
```

每轮实验开始前必须记录：

```powershell
git status --short --branch
git rev-parse HEAD
```

把本轮开始时的 commit 写入 `实验记录.md` 的本轮实验计划。

回退未提升实验时，只恢复本轮代码/配置改动，不恢复 `PLAN.md`：

```powershell
git restore baseline_lgb.py train_catboost.py blend_predictions.py environment.yml README.md "实验记录.md"
```

如果本轮新增了失败实验脚本，且确认不保留，再删除对应脚本。删除前必须用 `git status --short` 确认文件名。

保留实验时提交并推送：

```powershell
git add baseline_lgb.py train_catboost.py blend_predictions.py environment.yml README.md "实验记录.md" PLAN.md
git commit -m "Improve local AUC to NEW_AUC with EXP_NAME"
git push -u origin goal-auc-075
```

## 5. Fixed Validation Commands

验证当前最佳 AUC：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python -c "import json; print(json.load(open(r'outputs_blend_exp012/metrics_blend_exp012_lgb_cat_grid001.json'))['auc'])"
```

复算当前最佳融合：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python blend_predictions.py --oof outputs_blend_exp009/oof_blend_exp009_equal.csv outputs_cat_exp012/oof_cat_exp012.csv --sub outputs_blend_exp009/submission_blend_exp009_equal.csv outputs_cat_exp012/submission_cat_exp012.csv --search-step 0.01 --output-dir outputs_blend_recheck_best --run-name blend_recheck_best
```

submission 格式验证。把 `OUTPUT_SUBMISSION_PATH` 替换为候选 submission：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python -c "import pandas as pd; p=r'OUTPUT_SUBMISSION_PATH'; s=pd.read_csv(p); print(s.shape); print(s.columns.tolist()); print(s['isDefault'].min(), s['isDefault'].max(), s['isDefault'].mean()); assert s.shape==(200000,2); assert s.columns.tolist()==['id','isDefault']; assert s['isDefault'].between(0,1).all()"
```

读取指标文件时必须确认字段名。LightGBM/CatBoost 通常使用 `cv_auc`，融合脚本使用 `auc`。

## 6. Required Loop

每一轮必须按以下顺序执行：

1. 每轮实验必须先在 `实验记录.md` 追加本轮实验计划，再修改代码或运行训练。
2. 只做一个明确实验，不要同时混合多个方向。
3. 先跑 smoke test。
4. smoke test 达到门槛后，才允许跑全量 5 折。
5. 全量结果出来后，必须和当前最佳进行融合或直接比较。
6. 验证候选 submission 格式。
7. 按第 10 节保留/回退规则处理。
8. 更新 `实验记录.md`。

## 7. Smoke Gate

CatBoost smoke test 门槛：

```text
smoke OOF AUC >= 0.7225
```

LightGBM smoke test 门槛：

```text
smoke OOF AUC >= 0.7190
```

未达到 smoke 门槛，不跑全量。只记录结果并进入下一实验。

## 8. Experiment Priority

### EXP013-EXP015: CatBoost 参数微调

优先执行，因为当前最大提升来自 CatBoost 原生类别处理。

smoke 命令模板：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python train_catboost.py --sample-rows 50000 --n-splits 3 --iterations 700 --learning-rate LR --depth DEPTH --l2-leaf-reg L2 --early-stopping-rounds 100 --output-dir outputs_cat_EXP_smoke --run-name cat_EXP --numeric-category-cols
```

按顺序尝试：

```text
EXP013: depth=7, learning_rate=0.04, iterations=2200, l2_leaf_reg=10
EXP014: depth=6, learning_rate=0.035, iterations=2600, l2_leaf_reg=10
EXP015: depth=8, learning_rate=0.035, iterations=2200, l2_leaf_reg=12
```

全量命令模板：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python train_catboost.py --iterations ITERS --learning-rate LR --depth DEPTH --l2-leaf-reg L2 --early-stopping-rounds 180 --output-dir outputs_cat_EXP --run-name cat_EXP --numeric-category-cols
```

融合命令模板：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python blend_predictions.py --oof outputs_blend_exp012/oof_blend_exp012_lgb_cat_grid001.csv outputs_cat_EXP/oof_cat_EXP.csv --sub outputs_blend_exp012/submission_blend_exp012_lgb_cat_grid001.csv outputs_cat_EXP/submission_cat_EXP.csv --search-step 0.01 --output-dir outputs_blend_EXP --run-name blend_EXP
```

### EXP016: CatBoost 类别组合

只有 EXP013-EXP015 没有达到保留目标时才执行。

smoke：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python train_catboost.py --sample-rows 50000 --n-splits 3 --iterations 700 --learning-rate 0.04 --depth 7 --l2-leaf-reg 10 --early-stopping-rounds 100 --output-dir outputs_cat_exp016_smoke --run-name cat_exp016 --numeric-category-cols --category-combos
```

full：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python train_catboost.py --iterations 2200 --learning-rate 0.04 --depth 7 --l2-leaf-reg 10 --early-stopping-rounds 180 --output-dir outputs_cat_exp016 --run-name cat_exp016 --numeric-category-cols --category-combos
```

### EXP017: LightGBM 差异化补充

只有 CatBoost 连续无明显提升时才执行。不要大改特征工程，优先尝试与 CatBoost 差异较大的 LightGBM 参数。

候选方向：

```text
num_leaves=128
min_child_samples=80
learning_rate=0.025
n_estimators=3500
reg_alpha=0.3
reg_lambda=1.5
target_encoding=true
category_combos=false
```

smoke：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py --sample-rows 50000 --n-splits 3 --n-estimators 700 --learning-rate 0.025 --num-leaves 128 --min-child-samples 80 --reg-alpha 0.3 --reg-lambda 1.5 --subsample 0.85 --colsample-bytree 0.85 --early-stopping-rounds 50 --output-dir outputs_exp017_smoke --run-name lgb_exp017 --target-encoding
```

full：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py --n-estimators 3500 --learning-rate 0.025 --num-leaves 128 --min-child-samples 80 --reg-alpha 0.3 --reg-lambda 1.5 --subsample 0.85 --colsample-bytree 0.85 --early-stopping-rounds 120 --output-dir outputs_exp017 --run-name lgb_exp017 --target-encoding
```

### XGBoost

只有上面所有方向都无明显提升时才考虑 XGBoost。进入此阶段前，先在 `实验记录.md` 写明原因。

允许安装：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe install -n tianchi_finance -c conda-forge xgboost -y
```

安装后必须更新 `environment.yml`。新增脚本名固定为 `train_xgboost.py`，输出格式必须和现有脚本一致。

## 9. Blending Rules

每个全量模型至少做一次与当前最佳的二模型融合。

如果新增模型单模型 AUC 不高，但融合 AUC 提升，按融合 AUC 判定。

不要用 testA 的预测分布手动调权；权重只能由 OOF AUC 搜索或明确命令给出。

## 10. Keep Or Revert

设：

```text
best_auc = 0.7436926207152882
```

判定规则：

```text
candidate_auc >= 0.75:
  达成目标。更新 实验记录.md，提交并推送，停止继续实验。

candidate_auc >= best_auc + 0.0005:
  明显提升。更新 best_auc、best_submission、best_metrics，提交并推送。

best_auc < candidate_auc < best_auc + 0.0005:
  只记录，不更新最佳，不提交。可以保留输出文件用于后续融合，但不要推送。

candidate_auc <= best_auc:
  记录失败原因。回退本轮代码/配置改动，不提交。
```

如果多轮后产生新的最佳，后续所有比较必须使用新的 `best_auc`，并在 `实验记录.md` 顶部或最新实验记录中维护：

```text
Current best AUC:
Current best submission:
Current best metrics:
Current best commit:
```

## 11. 实验记录.md Template

每轮实验开始前，先在 `实验记录.md` 追加本轮实验计划。

每轮追加：

```markdown
## EXPxxx: short title

- Start commit:
- Current best before experiment:
- Goal:
- Files changed:
- Smoke command:
- Smoke AUC:
- Smoke decision:
- Full command:
- Full OOF AUC:
- Blend command:
- Blend AUC:
- Submission check:
- Candidate submission:
- Decision:
- Revert or commit:
```

## 12. Stop Conditions

满足任一条件即停止：

```text
1. local OOF AUC >= 0.75
2. 连续 5 个完整实验没有任何 candidate_auc > best_auc
3. 用户要求暂停
```

停止时输出：

```text
best_auc
best_submission
best_metrics
branch
commit
pushed_or_not
next_recommended_official_submission
```

当前不建议消耗官网提交次数。只有本地结果达到 `0.75`，或用户明确同意，才准备官网提交文件。
