# DineLens · 餐前点评

DineLens 用于分析中文餐饮评论。它识别 ASAP 定义的 18 个评价方面，判断各方面的情感，并用 TF-IDF 和 TextRank 定位负面评价的原文证据。

| 内容 | 位置 |
| --- | --- |
| 模型训练、推理与证据抽取 | [aspect_detection/](aspect_detection/) |
| ASAP 原始训练、验证和测试集 | [dataset/](dataset/) |
| 360 条证据补标样本 | [annotations/](annotations/) |
| AutoDL 实验脚本、运行配置、指标和逐条预测 | [runs/autodl-20261123/](runs/autodl-20261123/) |
| 网页及本地 API | [web/](web/) |
| 六张报告图及源数据 | [reports/figures/](reports/figures/) |
| 已训练模型 | [GitHub Release](https://github.com/Icey18899877399/DineLens/releases/tag/v0.1.0) |

### 数据来源

数据来自美团发布的 [ASAP 数据集](https://github.com/Meituan-Dianping/asap)，保留原始划分及 [Apache 2.0 许可](dataset/LICENSE)。官方训练集、验证集和测试集分别包含 36,850、4,940 和 4,940 条评论。引用数据请使用 [Bu 等人的 NAACL 2021 论文](https://aclanthology.org/2021.naacl-main.167/)。

### 下载已训练模型

从 [Release](https://github.com/Icey18899877399/DineLens/releases/tag/v0.1.0) 下载 `baseline-models.zip` 和 `joint-bert-checkpoint.zip`，将压缩包解压到项目根目录。压缩包保留了代码读取的 `runs/` 目录结构，Release 的 `SHA256SUMS.txt` 可用于核验下载。

- `baseline-models.zip` 包含传统基线模型。网页查看原文证据时使用方面分类基线中的 TF-IDF 向量器。
- `joint-bert-checkpoint.zip` 包含在 ASAP 上训练的联合 BERT、分词器及配置。网页的「分析我的评论」需要该模型和 `requirements-bert.txt` 中的依赖，使用 CPU 推理。
- 重新训练时，原始预训练权重可从 [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) 获取。AutoDL 运行的模型版本和环境记录在 [experiment_report.json](runs/autodl-20261123/runs/experiment_report.json)。

## 功能 1 · 方面多标签分类

该模块识别 ASAP 的全部 18 个方面。原标注 `-2` 对应未提及，`-1`、`0`、`1` 对应已提及。同一评论可命中多个方面。

- **传统基线**使用 Jieba、词级一元／二元 TF-IDF 和 18 个逻辑回归分类器。
- **BERT** 每条评论编码一次，通过 18 个独立的 sigmoid 分数预测提及。默认使用 `google-bert/bert-base-chinese`，也支持本地模型目录。
- 训练集拟合参数，验证集按每个方面的 F1 选择阈值。BERT 使用验证集 Macro-F1 选择训练轮次。测试集用于最终评估。
- 读取时按 Unicode 规范化、移除空白和忽略大小写识别重复评论。测试集原样保留，优先保留验证集，训练集移除与它们重复的文本。训练与验证内部保留首条。官方 CSV 保持原样。

## 在项目根目录运行

已有的本地环境可直接运行。新环境使用 Python 3.11 或 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m aspect_detection train --model baseline
.\.venv\Scripts\python.exe -m aspect_detection predict --run runs/aspect_detection/baseline --text "菜很好吃，但点完菜等了四十分钟才上桌。"
```

GPU 机器先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与驱动匹配的 CUDA 版 PyTorch，再安装其他依赖并训练。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-bert.txt
.\.venv\Scripts\python.exe -m aspect_detection train --model bert --device cuda
```

默认训练 3 轮，每批 8 条，最长 512 个 token，超长输入保留开头。显存紧张时先降低 `--batch-size`。预训练模型可通过 `--checkpoint` 指向含分词器、配置及 safetensors 权重的本地目录。重跑时用 `--output` 指定新的运行目录。

## 输出和网页接入

默认输出到 `runs/aspect_detection/<model>/`。

| 文件 | 用途 |
| --- | --- |
| `test_metrics.json` | 18 个方面的精确率、召回率、F1、人工提及数，以及 Macro-F1、Micro-F1 |
| `test_predictions.csv` | 一条评论 18 行，含 `id`、官方方面键、中文组名、人工提及结果、预测结果、分数和阈值 |
| `model.json` | 方面键顺序、验证集阈值、分割及训练信息 |
| `baseline.joblib` 或 `checkpoint/` | 推理所需的本地模型 |

网页可先加载一次本地模型，再对每次提交的评论调用 `predict()`。其 18 条返回值包含 `aspect`、`group`、`name`、`mentioned`、`score` 和 `threshold`。按 `group` 分五组，用 `mentioned` 控制已提及状态。

```python
from pathlib import Path
from aspect_detection.pipeline import load_predictor

predictor = load_predictor(Path("runs/aspect_detection/baseline"))
rows = predictor.predict("菜很好吃，但点完菜等了四十分钟才上桌。")
```

## 功能 2 · 方面情感判断

功能 2 联合输出 18 个方面的提及和情感。传统基线共用 Jieba 与 TF-IDF 特征，对每个方面在人工已提及的样本上拟合情感逻辑回归。BERT 的一次评论编码输出 18 个提及分数和 18 组情感分数。训练时，对人工未提及的方面屏蔽情感损失。用验证集情感 Macro-F1 选择 BERT 轮次。

```powershell
.\.venv\Scripts\python.exe -m aspect_detection train --task sentiment --model baseline
.\.venv\Scripts\python.exe -m aspect_detection train --task sentiment --model bert --device cuda
.\.venv\Scripts\python.exe -m aspect_detection predict --run runs/aspect_sentiment/baseline --text "菜很好吃，服务态度一般，但点完菜等了四十分钟才上桌。"
```

功能 2 默认输出到 `runs/aspect_sentiment/<model>/`。可通过 CLI 或原有的 `load_predictor()` 加载。每个方面保留功能 1 的字段，并增加 `sentiment`、`sentiment_name` 和 `sentiment_scores`。预测为未提及的方面以 `-2` 返回，情感分数为 `null`。

- `test_metrics.json` 顶层的 Macro-F1 等字段仍表示方面识别结果。`sentiment` 提供独立的三分类 Macro-F1、逐类及逐方面指标和混淆矩阵。混淆矩阵第 0、1、2 行列依次对应负面、中性、正面，行是真实类别。
- `test_predictions.csv` 每条评论仍有 18 行。`gold_sentiment` 保留 ASAP 原码（负面 -1、中性 0、正面 1、未提及 -2）。`sentiment` 供网页及排行使用，受预测提及门控。`sentiment_if_mentioned` 和三列情感分数用于实验核验，不受提及预测影响。
- 情感指标仅计算人工已提及的「评论＋方面」，其中也包含被方面识别层漏掉的项目。网页可直接读取混淆矩阵，并按方面筛选。

## 功能 3 · 负面评价证据抽取

功能 3 从评论原文返回连续片段，覆盖 ASAP 的 18 个方面。按标点和 ASAP 原文中的换行转义切分，保留原始字符位置。筛选时优先采用能直接确定方面的抱怨短语，例如「找不到车位」。对「太慢」这类通用负面表述，结合片段内最近的方面词判断。当前一个子句提供方面名称、后一个子句表达负面时，可以跨一个逗号补齐上下文。两种算法对同一候选集合排序。

- **TF-IDF**用目标方面及词表构成查询，按查询与候选的余弦相似度排序。向量器来自功能 1 的训练集，无需重新拟合。
- **TextRank**以候选片段作节点，片段之间的 TF-IDF 余弦相似度作边权，以目标方面相关性作为 PageRank 的跳转偏好。两种算法的分数采用各自的量纲。
- 匹配足够的方面和负面线索时才返回证据。比较、反讽、混合评价和跨句表达仍可能有错例。

```powershell
# 单独给定方面，比对两种排序
.\.venv\Scripts\python.exe -m aspect_detection evidence --aspect "Service#Timely" --text "菜好吃，但点完菜等了四十分钟才上桌。" --top-k 3

# 从功能 2 的负面预测出发，自动附上两个方法的首条证据
.\.venv\Scripts\python.exe -m aspect_detection predict --run runs/aspect_sentiment/baseline --text "菜好吃，但点完菜等了四十分钟才上桌。" --evidence
```

其他 TF-IDF 基线可通过 `--tfidf-run` 指定。网页启动时可以先加载一次抽取器，将功能 2 的预测交给 `annotate_predictions()`。每个方法的证据包含 `text`、分数和半开区间。Python 用 `review[start:end]`，JavaScript 用 `review.slice(start_utf16, end_utf16)`，后者可在含表情的评论里正确定位。无可定位片段时对应方法为 `null`。

```python
from pathlib import Path
from aspect_detection.evidence import load_evidence_extractor
from aspect_detection.pipeline import load_predictor

predictor = load_predictor(Path("runs/aspect_sentiment/baseline"))
extractor = load_evidence_extractor(Path("runs/aspect_detection/baseline"))
review = "🍜菜好吃，但点完菜等了四十分钟才上桌。"
rows = extractor.annotate_predictions(review, predictor.predict(review))
```

### 补标和比较

`annotations/asap_evidence_test.csv` 是从官方测试集按 18 个方面各抽 20 条负面评论生成的表。编辑时将 `id` 当作文本，保留方面键和原评论。

- `label` 填 `evidence` 时，给 `evidence_1` 填一段可以直接在原评论唯一找到的文本。可在 `evidence_2` 加入另一段可接受证据。
- 没有可单独定位的证据时，`label` 填 `none`，证据栏留空。尚未标注的行三个字段都留空。
- 若要在验证集试调线索，可以用 `evidence-sample --split dev --output annotations/asap_evidence_dev.csv` 生成独立标注表。

```powershell
.\.venv\Scripts\python.exe -m aspect_detection evidence-evaluate --annotations annotations/asap_evidence_test.csv --output runs/evidence/test_evaluation.json
```

评估只读取已完成行。对有证据样本报告 Hit@1 与字符级区间 F1。首条预测必须完整覆盖一个可接受片段且字符 F1 至少为 0.5 才算命中，未返回证据计为漏掉。对 `none` 样本报告误高亮率。结果 JSON 同时保存整体、逐方面和逐条对照结果，方便网页显示错例。

## 餐前点评网页

安装 `requirements.txt` 的依赖后，从项目根目录启动本地预览。使用「分析我的评论」前，解压模型并安装 `requirements-bert.txt` 的依赖。

```powershell
.\.venv\Scripts\python.exe -m web.server --port 8765
```

在浏览器打开 `http://127.0.0.1:8765/`。概览和评论筛选读取 AutoDL 带回的 4,940 条测试预测，点击评论可查看两种方法定位的原文。「分析我的评论」在本机按需加载联合 BERT 和抽取器。网页的「数据说明」包含模型总览、18 方面 F1 成对条形图、可切换模型的单张情感混淆矩阵和证据覆盖图。点击方面或矩阵格子可查看对应数值。六张独立报告图件和对应源数据保存在 `reports/figures/`。

首页照片由 [Anh Tuan To](https://unsplash.com/photos/a-close-up-of-a-delicious-meal-on-a-table-hGeGkdSOkCg) 拍摄，使用 [Unsplash 许可](https://unsplash.com/license)。标题字体为 Noto Serif SC，随项目附带字体许可。

## 核验

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

BERT 测试在具备可选依赖时运行，使用测试现场创建的小模型，覆盖实际训练、保存和重新加载。
