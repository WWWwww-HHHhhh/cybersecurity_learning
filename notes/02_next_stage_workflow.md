# 去重之后：完成数据探索，跑通第一版基线

这份记录接着 `01_data_exploration.ipynb` 往下做，对应十天计划的 Day 2 后半段到 Day 4。代码块按顺序放进 `02_data_preparation.ipynb`，从头运行即可，不依赖另一个 notebook 留在内存里的变量。原始 CSV、README 和已有 notebook 都不需要改。

我目前已经核对过：原文件为 461,043 行、45 列，11,071 条完全重复记录都属于 `normal`。去重后有 449,972 行，其中正常 288,929 行、攻击 161,043 行。以下代码会重新算出这些数，不把这段文字当作运行结果。

这不是对论文的严格复现。数据来自已记录的公开镜像；本轮选 100,000 行做可管理的实验，保留固定种子，并增加一项论文式随机划分之外的特征组合隔离检查。

## 1. 从原文件得到去重副本

先从同一份原始数据开始，避免 notebook 之间的变量状态影响结果。`drop_duplicates()` 只生成内存中的副本，不写回 `data/raw/`。

```python
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
data_path = root / "data" / "raw" / "Train_Test_Network.csv"
assert data_path.is_file(), f"找不到数据文件：{data_path}"

with data_path.open("rb") as file:
    print("SHA-256：", hashlib.file_digest(file, "sha256").hexdigest())

df = pd.read_csv(data_path, low_memory=False)
df_dedup = df.drop_duplicates()
print("原始行列数：", df.shape)
print("去重后行列数：", df_dedup.shape)
print(df_dedup["label"].value_counts().sort_index())
```

我核对到的哈希是 `65d5465df1809b984fd10e4703bc9c012d1aefd11803773856364216f0a3520d`，去重后应为 `(449972, 45)`。如果不一致，先停下来核对文件，不继续模型实验。数目一致后，再看尚未解释的 `weird` 字段。

## 2. 单独看 `weird`，不把它当成攻击标签

`weird` 记录的是 Zeek 解析流量时遇到的异常情况，不是一种协议，也不等于攻击。我先看它有多常见、出现在哪些类别，以及 `weird_notice` 实际记录了什么。

```python
has_weird = df_dedup["weird_name"].ne("-")

print("有 weird_name 的行数：", has_weird.sum())
print("这些行的类别：")
display(df_dedup.loc[has_weird, "type"].value_counts())
print("weird_name 的取值：")
display(df_dedup.loc[has_weird, "weird_name"].value_counts())
print("这些行的 weird_notice：")
display(df_dedup.loc[has_weird, "weird_notice"].value_counts(dropna=False))
```

我在去重后的文件中看到 1,294 行有 `weird_name`，其中 1,291 行为正常流量，3 行为 DDoS。对应的 `weird_notice` 全是 `F`。因此我不会写“出现 weird 就是攻击”。后续只保留 `weird_name` 是否出现的标志，不把它的具体名称当作攻击类别，也不把几乎重复这个标志的 `weird_notice` 再放进模型。接下来检查数值字段，避免只看标签和文本列。

## 3. 用少量有意义的图表结束初步 EDA

我关心的是类别比例，以及连接时长和源端字节数的分布。先看极值，再画两张类别图和两张流量图。流量图每类各抽 5,000 行，只比较形状，不用于估计总体比例。

```python
import matplotlib.pyplot as plt
import seaborn as sns

core = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"]
audit = pd.DataFrame({
    "最小值": df_dedup[core].min(),
    "中位数": df_dedup[core].median(),
    "99分位": df_dedup[core].quantile(0.99),
    "最大值": df_dedup[core].max(),
    "负值数": df_dedup[core].lt(0).sum(),
})
display(audit)

view = df_dedup.groupby("label", group_keys=False).sample(n=5000, random_state=42)
view = view.assign(
    log_src_bytes=np.log1p(view["src_bytes"]),
    log_duration=np.log1p(view["duration"]),
)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df_dedup["label"].value_counts().sort_index().plot.bar(ax=axes[0, 0], title="Normal / attack")
df_dedup["type"].value_counts().plot.bar(ax=axes[0, 1], title="Attack types")
sns.histplot(data=view, x="log_src_bytes", hue="label", bins=40,
             stat="density", common_norm=False, ax=axes[1, 0])
sns.histplot(data=view, x="log_duration", hue="label", bins=40,
             stat="density", common_norm=False, ax=axes[1, 1])
fig.tight_layout()
plt.show()
```

我核对到这些核心数值列没有负值，但有长尾，例如 `src_bytes` 的 99 分位数为 3,016，最大值超过 38 亿。图里使用 `log1p` 只是为了看清分布，**没有修改训练数据**。不要凭极值就删行；把长尾记为后续 PCA 的注意点即可。至此初步 EDA 已能支撑下一步，不继续漫无目的地逐列翻看。

## 4. 固定一个可复现的实验子集

十天计划允许在本机资源范围内取一个子集。我先按 `type` 分层抽 100,000 行，使人数较少的 MITM 也保留。`type` 只用于抽样和解释，不会进入模型。

```python
from sklearn.model_selection import train_test_split

sample_index, _ = train_test_split(
    df_dedup.index,
    train_size=100_000,
    stratify=df_dedup["type"],
    random_state=42,
)
work = df_dedup.loc[sample_index].sort_index()
print("实验子集行列数：", work.shape)
display(work["type"].value_counts())
```

这份文件在同一版本的 pandas、scikit-learn 和同一随机种子下应得到 100,000 行，其中 MITM 约 232 行。这里的抽样属于本项目的算力取舍，不应写成论文使用了同一子集。下一步才决定哪些字段能作为输入。

## 5. 明确输入字段，先挡住容易看见的泄漏

论文去掉 `ts`、源/目标 IP 和端口这五列，避免模型记住测试环境标识。我也去掉 `type`，因为它直接说明攻击类别。`label` 是要预测的答案。高基数文本如 DNS 查询和 HTTP URI 暂时只记录“有没有值”，与论文对这类字段采用二值编码的思路接近，但不是逐项复现。

```python
availability_cols = [
    "dns_query", "ssl_subject", "ssl_issuer", "http_uri",
    "http_user_agent", "http_orig_mime_types",
    "http_resp_mime_types", "weird_name",
]
drop_cols = ["label", "type", "ts", "src_ip", "src_port",
             "dst_ip", "dst_port", "weird_notice"]

X = work.drop(columns=drop_cols).copy()
for col in availability_cols:
    X[col + "_present"] = X[col].ne("-").astype("int8")
X = X.drop(columns=availability_cols)
y = work["label"]

print("输入形状：", X.shape)
print("目标类别：", y.value_counts().to_dict())
print("文本列：", X.select_dtypes(include=["str", "object"]).columns.tolist())
assert "label" not in X and "type" not in X
```

应得到 37 列原始/派生输入。此时只是确定输入形式，尚未用全数据拟合编码器或缩放器。`-` 在剩余低基数类别列里先保留为一个类别，不全局当作 0。接下来先按十天计划和论文做 80/20 分层划分，再检查这种划分的局限。

## 6. 做论文式分层划分，并量化相同特征组合的重叠

普通分层划分让训练和测试的正常/攻击比例接近，也便于对照论文。不过，完全重复行去掉后，去掉环境标识列可能使不同记录拥有相同的模型输入。所以我会明确数一遍测试集中有多少条特征组合已经在训练集出现过。

```python
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
train_hash = pd.util.hash_pandas_object(X_train_s, index=False)
test_hash = pd.util.hash_pandas_object(X_test_s, index=False)
overlap = test_hash.isin(set(train_hash)).sum()

print("分层划分：", len(X_train_s), "训练 /", len(X_test_s), "测试")
print("两边攻击比例：", y_train_s.mean(), y_test_s.mean())
print("测试集中与训练集特征指纹相同的行数：", overlap)
```

我用上述固定版本和种子预检时得到 80,000/20,000 行，其中 11,561 条测试行的特征指纹在训练集中出现过。哈希用来快速发现相同组合，不能替代人工解释这些流量为何相同。这样的随机划分仍可作为与论文接近的对照，但不能单独支撑“能识别未见过的流量模式”的结论。下一步加一份隔离特征组合的测试集。

## 7. 增加特征组合隔离划分

我只用输入特征的指纹分组，不按模型分数挑测试集。五折中选测试行数和攻击比例最接近目标的一折。这样相同特征组合不会同时进入训练与测试，但测试比例不一定恰好为 20%，所以要把实际比例打印出来。

```python
from sklearn.model_selection import StratifiedGroupKFold

groups = pd.util.hash_pandas_object(X, index=False)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(X, y, groups))
best = min(
    range(len(folds)),
    key=lambda i: abs(len(folds[i][1]) / len(X) - 0.20)
                  + abs(y.iloc[folds[i][1]].mean() - y.mean()),
)
train_pos, test_pos = folds[best]
X_train_g, X_test_g = X.iloc[train_pos], X.iloc[test_pos]
y_train_g, y_test_g = y.iloc[train_pos], y.iloc[test_pos]

print("隔离划分：", len(X_train_g), "训练 /", len(X_test_g), "测试")
print("两边攻击比例：", y_train_g.mean(), y_test_g.mean())
print("跨边界共享的特征指纹数：",
      len(set(groups.iloc[train_pos]) & set(groups.iloc[test_pos])))
```

预检中选出约 81,228/18,772 行，跨边界共享的特征指纹为 0。它比普通划分更难，也更能检查模型面对未见特征组合时的表现；但它不等于真实部署环境，还要如实报告类别比例变化。两种划分都固定下来后，才拟合预处理和模型。

## 8. 用同一套规则跑 Original 特征基线

这一步只做 Original × 决策树/随机森林，不抢跑特征选择和 PCA。低基数文本由训练集拟合 One-Hot，数值列缺失填补也只在训练集拟合。计时把预处理包含在训练和预测过程里，以便后续方法用同一口径比较。

```python
from time import perf_counter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include=["str", "object"]).columns.tolist()
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                           dtype=np.float32), cat_cols),
], sparse_threshold=0)

models = {
    "DT": DecisionTreeClassifier(max_depth=20, min_samples_leaf=2,
                                 random_state=42),
    "RF": RandomForestClassifier(n_estimators=100, max_depth=20,
                                 min_samples_leaf=2, n_jobs=-1,
                                 random_state=42),
}
splits = {
    "stratified": (X_train_s, X_test_s, y_train_s, y_test_s),
    "group_holdout": (X_train_g, X_test_g, y_train_g, y_test_g),
}

rows = []
for split_name, (x_train, x_test, target_train, target_test) in splits.items():
    for model_name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocess", clone(preprocess)),
            ("model", clone(estimator)),
        ])
        start = perf_counter()
        pipeline.fit(x_train, target_train)
        train_seconds = perf_counter() - start
        start = perf_counter()
        predicted = pipeline.predict(x_test)
        predict_seconds = perf_counter() - start
        tn, fp, fn, tp = confusion_matrix(
            target_test, predicted, labels=[0, 1]
        ).ravel()
        rows.append({
            "split": split_name, "model": model_name,  
            "train_rows": len(x_train), "test_rows": len(x_test),
            "dimensions": len(pipeline.named_steps["preprocess"].get_feature_names_out()),
            "macro_f1": f1_score(target_test, predicted, average="macro"),
            "attack_recall": recall_score(target_test, predicted, pos_label=1),
            "weighted_f1": f1_score(target_test, predicted, average="weighted"),
            "accuracy": accuracy_score(target_test, predicted),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
        })

baseline = pd.DataFrame(rows)
display(baseline)
output = root / "results" / "baseline_split_comparison.csv"
if output.exists():
    print("结果文件已存在，未覆盖：", output)
else:
    baseline.to_csv(output, index=False)
    print("已保存：", output)
```

应得到四行：两种划分各有 DT、RF 一行。先核对 `fn` 和攻击类召回率，再看 Macro-F1，最后才谈准确率和速度。预检中普通划分的 Macro-F1 接近 0.99，隔离划分约为 0.76，攻击类召回率约为 0.51；这是划分方式造成的巨大差异，不能把普通划分的高分当作最终能力。正式结果以你运行并保存的 CSV 为准。

## 这一阶段何时算完成

哈希、行数和 `weird` 统计能复核，四张探索图与数值表能解释数据，`label`/`type`/环境标识没有进入输入，两种划分及其实际类别比例有记录，基线 CSV 有四行且混淆矩阵与指标一致。达到这些条件就停止 EDA，不再反复查看孤立字段。

后续按十天计划进入 Pearson/Chi-square 特征选择，再做与所选维度相同的 PCA。两种方法必须沿用这里固定的样本和划分，编码、缩放、特征选择及 PCA 都只能在训练集上拟合。主要比较以特征组合隔离划分为准，论文式分层划分作为对照；这个选择来自当前数据的重叠检查，并不是论文原本的实验设置。

方法参考：[Optimizing IoT intrusion detection system: feature selection versus feature extraction in machine learning](https://link.springer.com/article/10.1186/s40537-024-00892-y)，[Enhancing IoT security: A comparative study of feature reduction techniques for intrusion detection system](https://www.sciencedirect.com/science/article/pii/S2667305324000814)。
