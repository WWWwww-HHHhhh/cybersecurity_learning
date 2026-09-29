# 基于 Network TON-IoT 的入侵检测特征对比

这个项目使用 Network TON-IoT 的网络流量数据，比较原始特征、特征选择和 PCA 三种输入方式对入侵检测的影响。我想把问题做小、做完整，而不是一开始就堆很多模型

这个问题受到李靖老师关于物联网入侵检测和特征降维的研究启发，但本项目不是论文复现。目前还没有实验结果，也不预设哪种方法会赢

# 回答的问题

在使用同一份数据、同一组训练和测试划分、同样的预处理和分类器时，特征选择与 PCA 对准确率、攻击识别能力、训练效率和可解释性分别有什么影响？更具体地说，保留少量有明确含义的原始网络特征，与把特征压缩成主成分相比，哪一种更适合这个轻量级实验

# 假设

我目前猜测 特征选择可能保留更清楚的网络安全语义，也可能减少训练和推理时间。PCA 可能得到更紧凑的输入，并在某些模型中保持或改善检测表现，但主成分通常不如原始网络字段直观。这只是开始实验前的假设，结果可能支持它，也可能推翻它

# 实验大致流程

下载TON-IoT 的网络流量数据之后，先做正常流量与攻击流量的二分类。数据清洗后，在相同的训练集和测试集上比较三组输入：全部可用特征（Original）、通过 Pearson 冗余检查和 Chi-square 选择出的原始特征（Feature Selection），以及 PCA 得到的主成分。分类器先用决策树（Decision Tree），再用随机森林（Random Forest）检查结论是否稳定

主要看 Macro-F1 和攻击类召回率，同时记录准确率、Weighted-F1、混淆矩阵、训练时间、推理时间及保留的特征数量。不能只凭准确率判断，因为正常流量与攻击流量的数量可能不均衡。编码、标准化、特征选择和 PCA 都只能在训练集上拟合，测试集只用于评估。默认随机种子为 `42`

# 重要环节

- 数据：Network TON-IoT 的可追溯子集
- 任务：正常流量与攻击流量二分类
- 表示方法：Original、Feature Selection、PCA
- 模型：Decision Tree 和 Random Forest
- 主指标：Macro-F1、攻击类 Recall
- 辅助指标：Accuracy、Weighted-F1、混淆矩阵、训练时间和推理时间
- 实验约束：相同训练/测试划分，预处理只在训练集上拟合，默认随机种子为 42。

# 这次先不做什么

- BoT-IoT 或其他第二数据集
- CNN、Autoencoder 或深度迁移学习
- 遗传算法调参和集成学习系统
- 大规模超参数搜索
- 对相关论文结果的严格复现声明

# 产出

1. 降维后，攻击检测能力有没有明显变化
2. 特征选择与 PCA，哪一种在检测效果和耗时之间更平衡
3. 被选中的原始特征能否用网络安全知识解释
4. PCA 的效果变化是否值得牺牲一部分可解释性
5. 抽样、类别不平衡和只使用一个数据集，会怎样限制结论

# 目前进度

项目目录、Python 环境和一份 Network TON-IoT 网络流量 CSV 已经准备好。数据审计、抽样、实验代码和结果还没有完成。因此现在可以按下面的方法重建开发环境，但还不能复现尚未产生的实验结论。抽样记录和运行命令会随着实验落地补进来

# 环境与安装

我在 Windows 11 的 WSL2 Ubuntu 24.04 中开发，项目放在 Linux 文件系统里。使用 Miniconda 管理独立环境，Python 版本为 3.11.16。当前环境中已核验的直接依赖记录在 `requirements.txt`，包括 NumPy、pandas、scikit-learn、Matplotlib、seaborn、JupyterLab 和 ipykernel。Git 版本为 2.43.0。Python、Git、Ubuntu 和 Miniconda 不属于 pip 依赖，因此不写进 `requirements.txt`

在已安装 Miniconda 的 WSL Ubuntu 中，可以从项目根目录安装已记录的 Python 版本和直接依赖：

```bash
cd ~/projects/cybersecurity_learning
conda create -n cybersecurity_learning python=3.11.16
conda activate cybersecurity_learning
python -m pip install -r requirements.txt
```

如果已有同名环境，不要重复创建，先用 `conda env list` 检查。安装完成后可用 `python --version` 和 `python -m pip list` 核对版本。JupyterLab 已包含在依赖中，需要打开笔记本时运行 `jupyter lab`


# 目录

```text
cybersecurity_learning/
├── README.md                 # 项目问题、实验思路和运行环境
├── requirements.txt          # Python 依赖及版本
├── .gitignore                # 不提交的数据和临时文件规则
├── data/
│   ├── raw/                  # 保留下载的数据，不在这里做清洗
│   │   └── Train_Test_Network.csv
│   └── processed/            # 后续保存清洗或抽样后的数据
├── notebooks/                # 数据探索和实验笔记本
├── notes/                    # 从零开始的过程与实验决策记录
│   └── process.md
├── results/                  # 后续保存指标和对比结果
│   └── figures/              # 后续保存混淆矩阵等图表
├── scripts/                  # 后续放可重复运行的实验入口
└── src/                      # 后续放可复用的数据处理和评估代码
```
