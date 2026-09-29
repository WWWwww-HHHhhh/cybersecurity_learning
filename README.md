# 轻量级物联网入侵检测实验

这个项目打算使用 Network TON-IoT 的网络流量数据，比较原始特征、特征选择和 PCA 三种输入方式对入侵检测的影响。我想把问题做小、做完整，而不是一开始就堆很多模型。

这个问题受到李靖老师关于物联网入侵检测和特征降维的研究启发，但本项目不是论文复现。目前还没有实验结果，也不预设哪种方法会赢。

# 回答的问题

在使用同一份数据、同一组训练和测试划分、同样的预处理和分类器时，特征选择与 PCA 对攻击识别能力、运行效率和可解释性分别有什么影响？更具体地说，保留少量有明确含义的原始网络特征，与把特征压缩成主成分相比，哪一种更适合这个轻量级实验？

我目前猜测，特征选择可能更容易解释，也可能减少训练和推理时间。PCA 可能得到更紧凑的输入，但主成分通常不如原始网络字段直观。这只是开始实验前的假设，结果出来后再判断。

# 实验大致流程

计划使用 Network TON-IoT 的网络流量数据，先做正常流量与攻击流量的二分类。数据清洗后，在相同的训练集和测试集上比较三组输入：全部可用特征、选择出的原始特征，以及 PCA 得到的主成分。分类器先用决策树，再用随机森林检查结论是否稳定。

主要看 Macro-F1 和攻击类召回率，同时记录准确率、Weighted-F1、混淆矩阵、训练时间、推理时间及保留的特征数量。不能只凭准确率判断，因为正常流量与攻击流量的数量可能不均衡。编码、标准化、特征选择和 PCA 都只能在训练集上拟合，测试集只用于评估。默认随机种子为 `42`。

这轮实验暂时不扩展到第二个数据集、多分类、深度学习或大规模调参，也不会把结果称为对某篇论文的严格复现。完成后，我希望能说明降维是否影响攻击检测、两种方法在效果和耗时之间各有什么取舍、选出的原始特征能否用网络安全知识解释，以及单一数据集和抽样方式会怎样限制结论。

# 目前进度

项目目录和 Python 环境已经建立，数据还没有放入项目，实验代码和结果也还没有完成。因此现在可以按下面的方法重建开发环境，但还不能复现尚未产生的实验结论。数据文件、抽样记录和运行命令会随着实验落地补进来。

# 环境与安装

我在 Windows 11 的 WSL2 Ubuntu 24.04 中开发，项目放在 Linux 文件系统里。使用 Miniconda 管理独立环境，Python 版本为 3.11.16。当前环境中已核验的直接依赖记录在 `requirements.txt`，包括 NumPy、pandas、scikit-learn、Matplotlib、seaborn、JupyterLab 和 ipykernel。Git 版本为 2.43.0。Python、Git、Ubuntu 和 Miniconda 不属于 pip 依赖，因此不写进 `requirements.txt`。

在已安装 Miniconda 的 WSL Ubuntu 中，可以从项目根目录安装已记录的 Python 版本和直接依赖：

```bash
cd ~/projects/cybersecurity_learning
conda create -n cybersecurity_learning python=3.11.16
conda activate cybersecurity_learning
python -m pip install -r requirements.txt
```

如果已有同名环境，不要重复创建，先用 `conda env list` 检查。安装完成后可用 `python --version` 和 `python -m pip list` 核对版本。JupyterLab 已包含在依赖中，需要打开笔记本时运行 `jupyter lab`。

# 数据记录

计划使用 [TON-IoT 数据集](https://research.unsw.edu.au/projects/toniot-datasets) 中的 Network TON-IoT 数据。原始数据放在 `data/raw/`，处理后的数据放在 `data/processed/`。数据文件不会提交到 Git。

目前还没有选定具体文件。开始实验时需要在这里记录实际文件名、下载来源、抽样方法、抽样比例或样本数。默认随机种子为 `42`，如果实验中改动，也会同步记录。没有这些信息，就不能把结果称为可复现。

# 目录

```text
cybersecurity_learning/
├── README.md
├── requirements.txt
├── data/          # 原始数据和处理后的数据
├── notebooks/     # 数据检查与实验记录
├── src/           # 可复用的处理和评估代码
├── scripts/       # 实验运行入口
├── results/       # 结果表和图
└── notes/         # 研究问题和实验决策
```

# 相关研究

项目选题参考了李靖老师关于物联网入侵检测中特征选择、特征提取和 NetFlow 的研究。后续阅读论文时，我会在 `notes/` 里记录具体方法和引用，并区分论文结论与本项目自己的实验结果。
