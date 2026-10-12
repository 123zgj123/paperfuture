"""Draw the proposed six-stage PaperFuture framework, in Chinese and English.

Run: python3 figures/draw_paperfuture_framework.py
Requires matplotlib and a Chinese font for the Chinese edition. No API calls.
Writes editable-text SVG, vector PDF, and 240-dpi PNG next to this script.
These figures describe the design, not completed implementation or results.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paperfuture-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont, fontManager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as PlotPath


OUT = Path(__file__).resolve().parent
W, H = 1800, 1110
C = {
    "ink": "#203247", "muted": "#586C80", "border": "#C7D3DF",
    "blue": "#2D65A1", "blue_bg": "#F1F6FC",
    "teal": "#157B78", "teal_bg": "#EEF8F6",
    "purple": "#7561A4", "purple_bg": "#F5F2FA",
    "gold": "#95671E", "gold_bg": "#FCF6E9",
    "white": "#FFFFFF", "rail": "#F7F9FC",
}

COPY = {
    "zh": {
        "title": "PaperFuture",
        "subtitle": "从研究发展轨迹出发，模拟并评估未来技术跃迁",
        "domains": "研究领域：多模态大模型自进化中的可解释性  /  Agent security  /  AI4EDA",
        "api": "所有模型能力通过 API 调用",
        "prep": "A  历史资料准备 · 按领域、子方向与截止时间 T 构建并复用",
        "lit": "文献收集与筛选",
        "lit1": "分年采集，重点覆盖最近两年",
        "lit2": "词法 + 向量召回 → 精排与轨迹覆盖选择",
        "lit_out": "Selected data：版本化论文 + 可追溯原文片段",
        "graph": "带权有向发展图",
        "graph_lines": ["节点：阶段、论文与核心机制", "边：独立复核技术关系", "粗细：LLM 评分 0—100", "结构：分叉、汇合与未解瓶颈"],
        "graph_note": "边权重表示变化幅度，不代表成功概率",
        "graph_feed": "完整阶段图 + 本轮检索的原文证据",
        "loop": "B  动态探索 · 一个候选对应一轮 · 全局最多四轮",
        "leap": "LLM 选择跃迁",
        "leap_lines": ["从当前瓶颈与历史转折出发", "确定改变的机制或假设", "选择目标幅度，并说明依据"],
        "leap_out": "候选 idea + 条件假设",
        "world": "World model 模拟与讨论",
        "sim": "多模型独立模拟",
        "sim_sub": "成立 / 部分成立 / 失败，以及后续瓶颈",
        "critic": "独立评议与讨论",
        "critic_sub": "支持与异议均需依据；未知保留为假设",
        "merge": "综合判断",
        "merge_sub": "保留新颖机制，给出价值与可行性",
        "control": "动态探索决策",
        "decision": "足够好？\n或停滞 / 已四轮？",
        "stop": "停止",
        "continue": "继续",
        "control_note": "确定性规则\n不再生成新 idea",
        "result": "Idea 与价值判断",
        "result_items": [("主 idea", "问题、机制与新颖点"), ("价值与可行性", "成立条件与不确定性"), ("验证建议", "最小实验与来源依据")],
        "result_note": "选择已评估的最佳候选",
        "feedback": "全部前轮条件经验 + 保留假设 → 重检索原文并调整下一跳（t + 1）",
        "legend": "粗细不同的实线：历史技术变化    虚线：模拟反馈；不写回历史事实",
        "status": "v0.3 系统框架 · 模拟判断仍需真实实验验证",
    },
    "en": {
        "title": "PaperFuture",
        "subtitle": "Forecasting future research through trajectory-guided technical leaps",
        "domains": "Domains: interpretable multimodal self-evolution  /  Agent security  /  AI4EDA",
        "api": "All model inference via APIs",
        "prep": "A  Historical preparation · Reused for each domain, subfield and cutoff T",
        "lit": "Literature collection & selection",
        "lit1": "Year-stratified collection; recent two-year focus",
        "lit2": "Lexical + dense retrieval → rerank for coverage",
        "lit_out": "Selected data: versioned papers + source passages",
        "graph": "Weighted directed trajectory graph",
        "graph_lines": ["Nodes: stages, papers, mechanisms", "Edges: independently audited relations", "Width: LLM change score (0–100)", "Branches, merges, bottlenecks"],
        "graph_note": "Edge weight is change magnitude, not success probability",
        "graph_feed": "Global stage graph + round-specific source evidence",
        "loop": "B  Adaptive exploration · One candidate per round · At most four rounds in total",
        "leap": "LLM leap proposal",
        "leap_lines": ["Locate a bottleneck in the trajectory", "Specify the technical change", "Choose and justify leap magnitude"],
        "leap_out": "Idea + explicit assumptions",
        "world": "World model: simulate & discuss",
        "sim": "Independent model forecasts",
        "sim_sub": "Outcomes, failures and next bottlenecks",
        "critic": "Review and cross-model discussion",
        "critic_sub": "Evidence-backed objections; explicit unknowns",
        "merge": "Reconcile and assess",
        "merge_sub": "Preserve novelty; assess value and feasibility",
        "control": "Adaptive decision",
        "decision": "Good enough?\nStalled?\nRound 4?",
        "stop": "Stop",
        "continue": "Continue",
        "control_note": "Fixed decision rules\nNo new candidate generation",
        "result": "Idea & assessment",
        "result_items": [("Main idea", "Problem, mechanism, novelty"), ("Value & feasibility", "Conditions and uncertainty"), ("Validation plan", "Minimal test and sources")],
        "result_note": "Best assessed candidate",
        "feedback": "All prior conditional experiences + retained assumptions → retrieve evidence and choose next leap",
        "legend": "Solid edge width: historical change    Dashed loop: simulated feedback; never promoted to historical fact",
        "status": "v0.3 framework · Simulated assessments require empirical validation",
    },
}


def font(lang, *, bold=False):
    if lang == "zh":
        candidates = [
            os.environ.get("PAPERFUTURE_CJK_FONT", ""),
            "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                fontManager.addfont(candidate)
                family = FontProperties(fname=candidate).get_name()
                return FontProperties(family=["DejaVu Sans", family], weight="bold" if bold else "normal")
        raise RuntimeError("Set PAPERFUTURE_CJK_FONT to an installed Chinese font file")
    return FontProperties(fname=findfont(FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")))


def draw(lang):
    d = COPY[lang]
    normal, bold = font(lang), font(lang, bold=True)
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "svg.hashsalt": "paperfuture-framework-v1"})
    fig = plt.figure(figsize=(18, 11.1), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")

    def text(x, y, value, size=16.5, color=None, weight=False, align="left", va="top", z=5, max_width=None, **kwargs):
        artist = ax.text(x, y, value, fontsize=size, fontproperties=bold if weight else normal,
                         color=color or C["ink"], ha=align, va=va, linespacing=1.45, zorder=z, **kwargs)
        if max_width is not None:
            renderer = fig.canvas.get_renderer()
            allowed = max_width * fig.bbox.width / W
            measured = artist.get_window_extent(renderer=renderer).width
            if measured > allowed:
                artist.set_fontsize(size * allowed / measured)
        return artist

    def box(x, y, width, height, fill="white", border=None, radius=12, lw=1.2, z=1):
        patch = FancyBboxPatch((x, y), width, height,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              fc=fill, ec=border or C["border"], lw=lw, zorder=z)
        ax.add_patch(patch)
        return patch

    def arrow(points, color=None, width=1.8, dashed=False, z=2):
        path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points)-1))
        patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=16,
                                linewidth=width, color=color or C["ink"],
                                linestyle=(0, (5, 4)) if dashed else "-", zorder=z)
        ax.add_patch(patch)

    def card_title(number, x, y, title, color, size=20, max_width=None):
        ax.add_patch(Circle((x+22, y+22), 19, fc=color, ec="none", zorder=3))
        text(x+22, y+22, str(number), 16, C["white"], True, "center", "center")
        text(x+54, y+6, title, size, color, True, max_width=max_width)

    # Header and the three user-selected research domains.
    text(60, 26, d["title"], 31, C["blue"], True)
    text(400, 37, d["subtitle"], 20.5, max_width=1330)
    text(60, 87, d["domains"], 15.5, C["muted"])
    text(1740, 88, d["api"], 14, C["teal"], align="right")
    ax.plot([60, 1740], [127, 127], color=C["border"], lw=1)
    text(60, 146, d["prep"], 16, C["muted"], True)

    # 1. Literature collection and selection.
    box(60, 190, 725, 258, C["blue_bg"])
    card_title(1, 80, 208, d["lit"], C["blue"], 21, max_width=620)
    for i in range(3):
        px, py = 98 + 10*i, 302 - 10*i
        box(px, py, 51, 68, "white", C["blue"], radius=4, lw=1, z=2+i)
    for row in range(3):
        ax.plot([129, 158], [312+12*row, 312+12*row], color=C["blue"], lw=1.2, zorder=5)
    text(194, 282, d["lit1"], 18)
    text(194, 330, d["lit2"], 16.2, max_width=565)
    box(85, 389, 675, 40, C["white"], "#D8E3EF", radius=7)
    text(103, 400, d["lit_out"], 16, C["blue"], max_width=640)

    # 2. The evidence-backed stage graph; edge widths encode leap magnitude.
    box(850, 190, 890, 258, C["teal_bg"])
    card_title(2, 870, 208, d["graph"], C["teal"], 21, max_width=788)
    for i, line in enumerate(d["graph_lines"]):
        text(875, 273+31*i, line, 15.5, max_width=373)
    nodes = {"S1": (1298, 322), "S2": (1465, 275), "S3": (1465, 368), "S4": (1640, 322)}
    links = [("S1", "S2", 18), ("S1", "S3", 72), ("S2", "S4", 46), ("S3", "S4", 28)]
    for start, end, magnitude in links:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]
        arrow([(x0+33, y0), (x1-35, y1)], C["teal"], 0.9+magnitude/24, z=2)
        text((x0+x1)/2, (y0+y1)/2-14, str(magnitude), 13, C["teal"], align="center", va="center", bbox={"facecolor": C["teal_bg"], "edgecolor": "none", "pad": 1})
    for node, (x, y) in nodes.items():
        box(x-33, y-22, 66, 44, C["white"], C["teal"], radius=8, lw=1.3, z=3)
        text(x, y, node, 16, C["teal"], True, "center", "center")
    text(1298, 412, d["graph_note"], 12.8, C["muted"], align="center")
    arrow([(785, 316), (848, 316)])

    # The historical graph feeds the proposal, while evidence travels with it.
    arrow([(1284, 450), (1284, 491), (257, 491), (257, 578)], C["teal"], 2)
    text(786, 472, d["graph_feed"], 16, C["teal"], align="center", va="center",
         bbox={"facecolor": C["white"], "edgecolor": "none", "pad": 4})

    # Only the proposal, world-model and decision belong to the repeated loop.
    box(40, 527, 1365, 511, C["rail"], "#DDE4EE", radius=15, lw=1)
    text(510, 540, d["loop"], 15.2, C["muted"], True, max_width=872)

    # 3. A single proposed technical leap.
    box(60, 580, 395, 343, C["white"])
    card_title(3, 77, 599, d["leap"], C["blue"], 19.5, max_width=303)
    for i, line in enumerate(d["leap_lines"]):
        text(83, 680+47*i, line, 15.2, max_width=352)
    box(79, 844, 357, 53, C["blue_bg"], "#D8E3EF", radius=8)
    text(257, 870, d["leap_out"], 15.1, C["blue"], align="center", va="center", max_width=332)

    # 4. Roles are calls within the world-model stage, not extra system modules.
    box(510, 580, 555, 343, C["white"])
    card_title(4, 527, 599, d["world"], C["purple"], 18.6, max_width=465)
    for offset, (label, note) in enumerate([(d["sim"], d["sim_sub"]), (d["critic"], d["critic_sub"]), (d["merge"], d["merge_sub"])]):
        yy = 670 + 76*offset
        box(529, yy, 517, 65, C["purple_bg"], "#E5DFEF", radius=7)
        text(545, yy+7, label, 15.5, C["purple"], True)
        text(545, yy+34, note, 13.8, max_width=480)

    # 5. Deterministic stopping/continuation; global, never per branch.
    box(1120, 580, 265, 343, C["white"])
    card_title(5, 1135, 599, d["control"], C["gold"], 17.8, max_width=176)
    diamond = [(1252, 674), (1371, 744), (1252, 814), (1133, 744)]
    ax.add_patch(Polygon(diamond, closed=True, fc=C["gold_bg"], ec=C["gold"], lw=1.4, zorder=3))
    text(1252, 743, d["decision"], 13.2 if lang == "en" else 14.7, C["gold"], align="center", va="center")
    text(1252, 853, d["control_note"], 13.1, C["muted"], align="center", va="center", max_width=232)

    # 6. Final selection uses already assessed records, without a fifth idea.
    box(1440, 580, 300, 343, C["teal_bg"])
    card_title(6, 1453, 599, d["result"], C["teal"], 15.8, max_width=215)
    for i, (label, detail) in enumerate(d["result_items"]):
        yy = 675 + 66*i
        text(1460, yy, label, 16, C["teal"], True)
        text(1460, yy+29, detail, 14.4, max_width=260)
    text(1590, 898, d["result_note"], 12.8, C["muted"], align="center", max_width=267)

    arrow([(455, 745), (508, 745)])
    arrow([(1065, 745), (1118, 745)])
    arrow([(1385, 745), (1438, 745)])
    text(1413, 718, d["stop"], 12.5, C["gold"], align="center")

    # A single feedback return; simulated conditions do not alter historical G_T.
    arrow([(1252, 924), (1252, 1005), (257, 1005), (257, 924)], C["purple"], 2.1, dashed=True, z=4)
    text(1270, 949, d["continue"], 12.5, C["purple"])
    text(755, 972, d["feedback"], 15.3, C["purple"], align="center", va="center", max_width=950)

    text(60, 1053, d["legend"], 11.7, C["muted"])
    text(1740, 1081, d["status"], 11.5, C["muted"], align="right")

    stem = OUT / f"paperfuture_framework_{lang}"
    for extension in ("svg", "pdf", "png"):
        kwargs = {"dpi": 240} if extension == "png" else {}
        metadata = {"Creator": "PaperFuture framework drawing script"}
        if extension == "svg":
            metadata["Date"] = "2026-10-12"
        elif extension == "pdf":
            metadata["CreationDate"] = None
            metadata["ModDate"] = None
        fig.savefig(stem.with_suffix(f".{extension}"), facecolor="white", metadata=metadata, **kwargs)
        if extension == "svg":
            path = stem.with_suffix(".svg")
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(f"Created {stem.name}.svg / .pdf / .png")


if __name__ == "__main__":
    for language in ("zh", "en"):
        draw(language)
