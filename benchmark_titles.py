"""题目驱动 benchmark：把真实高校毕设题目灌入 topic_parser，统计系统能接住多少。

仅用解析层（parse_title），不跑实验，快速暴露「数据集匹配」这一核心瓶颈。
判定：
  - 可执行   ：matched_dataset 非 None 且任务与数据集一致
  - 误匹配   ：matched_dataset 非 None 但语义明显不符（如「医疗诊断」被错配到乳腺数据集）
  - 不可执行 ：matched_dataset 为 None（目录里没有对应数据）
"""
import sys, json
sys.path.insert(0, ".")
from engine.topic_parser import parse_title

# (来源, 题目) —— 均来自本次联网搜到的真实高校毕设题
TITLES = [
    # 深技大 / 通用计算机毕设（CSDN 选题库，贴近应用型本科）
    ("深技大风格", "基于深度学习的人脸识别系统"),
    ("深技大风格", "基于深度学习的车辆识别系统"),
    ("深技大风格", "基于深度学习的智能无人收银系统"),
    ("深技大风格", "基于聚类算法的用户分群分析"),
    ("深技大风格", "基于支持向量机的图像分类器"),
    ("深技大风格", "基于逻辑回归的信用评分模型"),
    ("深技大风格", "基于线性回归的房价预测模型"),
    ("深技大风格", "基于知识图谱的农业知识问答系统"),
    ("深技大风格", "基于深度学习的交通标志检测系统"),
    # 清华
    ("清华", "基于深度网络的RGBD图像分割算法研究"),
    ("清华", "基于机器学习的短时交通流预测算法的研究与实现"),
    ("清华", "基于XGBoost的网络游戏流失玩家预测算法研究"),
    ("清华", "基于社交网络结构的社会推荐算法研究"),
    # 北大（含优秀/十佳毕业论文）
    ("北大", "无监督学习范式下域迁移与重建技术研究"),
    ("北大", "图像和视频去雨的高效持续学习技术研究"),
    ("北大", "基于句法信息的文本语义相似性度量"),
    ("北大", "基于因果表示学习的时空分布预测"),
    ("北大", "图神经网络过平滑问题的研究"),
    ("北大", "阻尼随机牛顿法"),
    ("北大", "基于深度神经网络的智能视频计算"),
    ("北大", "基于检索增强的大语言模型算法研究"),
    ("北大工学院", "基于机器学习的复合材料等效弹性性能研究"),
    ("北大工学院", "基于迁移学习的模态检测方法"),
    # 深大
    ("深大", "城市轨道交通车站短时客流机器学习预测方法"),
    ("深大", "基于机器学习的入侵检测方法"),
    ("深大", "基于深度学习的车流量预测方法"),
    # 系统自身可覆盖的（对照）
    ("自测", "基于随机森林的心脏病预测研究"),
    ("自测", "基于梯度提升的翼型气动噪声预测研究"),
    ("自测", "基于支持向量机的鸢尾花分类研究"),
    ("自测", "基于机器学习的乳腺肿瘤诊断研究"),
    ("自测", "手写数字识别系统的设计与实现"),
    ("自测", "基于逻辑回归的糖尿病进展预测"),
    ("自测", "葡萄酒品种鉴别的机器学习方法对比"),
]

OK, MISS, WRONG = [], [], []

# 不可执行题目的归因：区分「能力缺口」与「题目本身不适合本引擎」
_REASONS = [
    ("视觉/视频后端", ["图像", "视频", "检测", "分割", "去雨", "标志", "车辆识别", "无人收银", "收银"],
     "需图像/视频模型（CNN 主干、检测头）；当前仅支持像素扁平化的小规模图像集"),
    ("文本/语言模型", ["语义", "文本", "自然语言", "问答", "大语言模型", "知识图谱", "检索增强"],
     "需文本语料与序列模型；当前文本仅支持 TF-IDF + 浅层分类"),
    ("图/网络结构", ["图神经网络", "社交网络", "图结构", "过平滑"],
     "需图数据与 GNN 后端，表格流水线无法表达节点-边结构"),
    ("时序/时空预测", ["时空", "客流", "车流", "交通流", "短时"],
     "需时序建模（滑窗、序列模型）；当前按独立样本建模，无法刻画时间依赖"),
    ("领域数据缺失", ["复合材料", "弹性", "入侵检测", "模态检测", "流失", "游戏"],
     "属真实领域数据题，需引入对应公开数据集（如 NSL-KDD、游戏流失数据）"),
    ("非机器学习题", ["牛顿法", "数值方法", "复杂度分析", "定理证明"],
     "纯数值/理论算法题，产出应是推导与复杂度分析，而非模型对比实验"),
]


def _diagnose(title: str):
    for name, kws, tip in _REASONS:
        if any(k in title for k in kws):
            return name, tip
    return "数据集目录未覆盖", "可在 DATASET_CATALOG 增补同类公开数据集，或用 --data 指定自有数据"


for i, (src, t) in enumerate(TITLES, 1):
    r = parse_title(t)
    ds = r["matched_dataset"]
    modality = r["matched_modality"]
    task = r["task_type"]
    hints = "、".join(r["method_hints"]) or "—"
    unsup = "、".join(r["unsupported_methods"]) or "—"
    if ds is None:
        verdict = "不可执行"
        MISS.append(t)
        reason, tip = _diagnose(t)
    else:
        # 模态一致性校验：题目点明的模态（图像/文本）必须与匹配数据集的 modality 一致，
        # 否则视为误匹配（如「图像题」被错配到表格集）。
        t_img = any(k in t for k in ["图像", "人脸", "面部", "像素", "img", "image", "face"])
        t_text = any(k in t for k in ["文本", "语义", "自然语言", "nlp", "语料", "文档", "新闻"])
        reason, tip = "—", "—"
        if t_img and modality != "image":
            verdict = "误匹配(应图像)"
            WRONG.append(t)
        elif t_text and modality != "text":
            verdict = "误匹配(应文本)"
            WRONG.append(t)
        else:
            verdict = "可执行"
            OK.append(t)
    print(f"{i:2d} [{src:6s}] {verdict:10s} | {task:12s} | mod:{str(modality):6s} | {str(ds):30s} | 方法:{hints:18s} | 不支持:{unsup}")
    print(f"     题目：{t}")
    if verdict != "可执行":
        print(f"     归因：{reason} —— {tip}")

print("\n================ 汇总 ================")
print(f"可执行(MATCH): {len(OK)}/{len(TITLES)}")
print(f"误匹配(语义错): {len(WRONG)}/{len(TITLES)}  -> {WRONG}")
print(f"不可执行(无数据集): {len(MISS)}/{len(TITLES)}")
print(f"可接住率: {round(100*len(OK)/len(TITLES),1)}%")
print("\n-- 不可执行的题目清单（按缺口归类）--")
_buckets = {}
for t in MISS:
    _buckets.setdefault(_diagnose(t)[0], []).append(t)
for name, items in sorted(_buckets.items(), key=lambda kv: -len(kv[1])):
    print(f"   [{name}] {len(items)} 题")
    for t in items:
        print("      ×", t)
