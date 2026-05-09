# Tianchi Finance Default Prediction

天池金融风控贷款违约预测项目。

## 项目目标

使用 `train.csv` 中的贷款与信用记录特征训练二分类模型，预测 `testA.csv` 中每个样本的违约概率 `isDefault`。比赛评测指标为 AUC。

## 当前基线

第一版 baseline 使用 LightGBM：

- 5 折 StratifiedKFold
- 基础日期、年限、比例和计数特征
- 本地 OOF AUC：0.737236
- 提交文件输出路径：`outputs/submission_lgb_baseline.csv`

详细实验记录见 `实验记录.md`。

## 环境准备

推荐使用 Miniforge/Conda：

```powershell
conda env create -f environment.yml
conda activate tianchi_finance
```

如果本机 `conda` 没有加入 PATH，可以使用完整路径，例如：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe env create -f environment.yml
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py
```

## 数据文件

数据文件不提交到 Git 仓库，需要放在项目根目录：

- `train.csv`
- `testA.csv`
- `sample_submit.csv`

## 运行 baseline

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py
```

快速冒烟测试：

```powershell
C:\Users\makab\Miniforge3\Scripts\conda.exe run -n tianchi_finance python baseline_lgb.py --sample-rows 50000 --n-splits 3 --n-estimators 300 --early-stopping-rounds 30 --output-dir outputs_smoke
```

