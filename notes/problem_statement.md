# 项目名称

轻量级 NetFlow IoT 入侵检测：基于 Network TON-IoT 的原始特征、特征选择与 PCA 对比

英文名称：Lightweight NetFlow IoT Intrusion Detection: Comparing Original Features, Feature Selection and PCA on Network TON-IoT
# 核心研究问题

在相同的数据划分、数据预处理和分类器条件下，特征选择和 PCA 对 Network TON-IoT 入侵检测的准确性、攻击识别能力、训练效率和可解释性分别有什么影响？

# 初步假设

特征选择可能保留更清楚的网络安全语义，并减少模型的训练与推理成本。PCA 可能得到更紧凑的表示，并在某些模型中保持或改善检测表现，但主成分通常不如原始字段容易解释。

这只是实验开始前的假设，不是结论。最终结果可能支持、部分支持或推翻它。

# 核心实验范围

- 数据：Network TON-IoT 的可追溯子集
- 任务：正常流量与攻击流量二分类
- 表示方法：Original、Feature Selection、PCA
- 模型：Decision Tree 和 Random Forest
- 主指标：Macro-F1、攻击类 Recall
- 辅助指标：Accuracy、Weighted-F1、混淆矩阵、训练时间和推理时间
- 实验约束：相同训练/测试划分，预处理只在训练集上拟合，默认随机种子为 42。

# 当前不包含

- BoT-IoT 或其他第二数据集
- CNN、Autoencoder 或深度迁移学习
- 遗传算法调参和集成学习系统
- 大规模超参数搜索
- 对相关论文结果的严格复现声明
- GitHub 推送

# 项目完成后希望能够回答

1. 降维之后，攻击检测能力是否发生明显变化
2. Feature Selection 与 PCA 哪一种在性能和耗时之间更平衡
3. 被选中的原始特征是否具有可以解释的网络安全含义
4. PCA 带来的性能变化是否值得牺牲解释性
5. 当前结论受到哪些抽样、类别不平衡或单数据集限制
