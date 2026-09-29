# 1. 选择 Network TON-IoT 数据

我想比较原始特征、特征选择和 PCA 对入侵检测的影响，所以先参考了《Optimizing IoT intrusion detection system: feature selection versus feature extraction in machine learning》。这篇论文使用的是 461,043 行的 Network TON-IoT 训练测试数据。我的实验重点是比较已有的网络流量特征，因此没有选择从原始 PCAP 重新提取特征，也没有改用 NetFlow 转换版。

我沿着 [UNSW 官方数据说明](https://research.unsw.edu.au/projects/toniot-datasets)找文件。最先拿到的 `train_test_network.csv` 只有 211,043 行、44 列，与[论文所述](https://link.springer.com/article/10.1186/s40537-024-00892-y)的规模不符，所以没有把它作为本次实验的主数据。我还查到一个早期官方 CloudStor 入口，但该服务已经停用；目前没有找到能直接核验为论文所用版本的官方文件。

后来我找到一个 [Hugging Face 公开镜像](https://huggingface.co/datasets/kunal0902/network-intrusion-iot/tree/main)。其中的 `Train_Test_Network.csv` 有 461,043 行、45 列，正常流量 300,000 行、攻击流量 161,043 行，与论文的关键统计一致。我核对了桌面和项目 `data/raw/` 中的文件，两者 SHA-256 相同，于是决定先用这份数据开展实验。

这份文件来自镜像，不能仅凭行数和类别分布证明它与论文原文件完全相同。因此我会注明数据来源，不把它写成“官方原版”，也不宣称严格复现论文。

