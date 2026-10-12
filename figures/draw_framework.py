"""Render the FarField research-framework proposal in Chinese and English.

Run with: python3 figures/draw_framework.py
Requires matplotlib; exports editable SVG, vector PDF, and 2x PNG.
The dashed components are planned extensions, not claims of completed work.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paperfuture-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, PathPatch
from matplotlib.path import Path as MplPath


OUT = Path(__file__).resolve().parent
W, H = 1920, 1180
C = {
    "ink": "#172B40", "muted": "#526477", "line": "#D6DEE7",
    "blue": "#3866AB", "blue_bg": "#F2F6FC",
    "teal": "#187D79", "teal_bg": "#EFF8F6",
    "amber": "#A26324", "amber_bg": "#FCF6EC",
    "purple": "#7860A3", "white": "#FFFFFF", "rail": "#F4F6F9",
}
CN_FONT = "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"

COPY = {
    "zh": {
        "title": "FarField：机制引导的远跳与分支内自进化",
        "subtitle": "新框架设计 · 从研究意图出发，通过真实反馈改善后续探索",
        "tag": "PAPERFUTURE / FRAMEWORK v0.1",
        "goal": "研究意图 + 初始材料",
        "goal_flow": "检查材料 → 暂定问题、瓶颈与未知量",
        "goal_note": "用途与约束保持稳定；科学子问题允许修订",
        "a": "机制引导的远跳", "a_sub": "GROUNDED REMOTE JUMP",
        "b": "OpenWorld：行动中认知", "b_sub": "PREDICT · ACT · OBSERVE · RESPOND",
        "c": "分支内自进化", "c_sub": "BRANCH-SCOPED MEMORY & SKILLS",
        "a1": "技术脉络与当前瓶颈",
        "a1_body": ["模型提出脉络草图；文献核验变化与边界"],
        "timeline": ["先前方法", "机制变化", "当前瓶颈"],
        "a2": "建立可检验的机制桥梁",
        "a2_body": ["来源机制 → 共享结构 → 目标方法", "明确关键假设、竞争解释与失效条件"],
        "a2_note": "优先探索条件性价值高、可检验的未知方向",
        "a3": "落点研究与区分实验",
        "a3_body": ["查相近方法、负面结果与适用边界", "提出廉价区分实验，保留开放实现空间"],
        "a3_note": "继续深入 / 修改落点 / 暂缓投入",
        "a_footer": "按需远跳或局部深化，继承前次实测发现",
        "b1": "提出预期与区分预测",
        "b1_body": ["记录候选解释、干预与实验前预测", "探索允许试探；确认前冻结评价协议"],
        "b2": "建设能力并真实执行",
        "b2_body": ["编写方法、接入数据、构建测量工具", "独立原生评价器 + 代码与原始回执"],
        "b2_note": "真实环境与观测 ≠ 模型内部推演",
        "b3": "核验测量并回应结果",
        "b3_body": ["检查实现、干预、测量与区分能力", "引用观测：维持 / 修订 / 追加验证 / 暂停"],
        "b3_note": "预测失配本身不能反驳科学假设",
        "performance": "方法性能", "mechanism": "机制证据",
        "separate": "分别验证",
        "c1": "有条件的方向记忆",
        "c1_body": ["保存发现、失败边界与未解决问题", "关联来源观测、适用条件和证据依赖"],
        "c2": "可执行技能版本",
        "c2_body": ["沉淀可复用的分析程序与实验流程", "局部试用 → 验证 → 复用 / 停用"],
        "c2_note": "进化研究能力；公共规则保持只读",
        "c3": "固定父分支快照后独立演化",
        "branch_parent": "父分支 · 学习包 vₜ",
        "branch_a": "方向 A · 记忆 / 技能",
        "branch_b": "方向 B · 记忆 / 技能",
        "c_footer": "只读材料可共享；分支学习互不覆盖",
        "candidate": "候选", "feedback": "反馈",
        "loop": "有来源与适用条件的记忆、技能 → 下一步问题、实验或跳跃",
        "retraction": "证据撤回 → 复查依赖经验与技能；有依据时允许维持判断",
        "base_title": "公共研究底座",
        "base": "论文与数据索引  ·  隔离执行  ·  独立评价  ·  原始事件与依赖  ·  统一预算",
        "outcome_title": "最终目标",
        "outcome": "发现值得研究的方向，获得独立验证的成果，并改善后续研究效用",
        "status_title": "实现边界",
        "status1": "实线：已有实现基础，仍需加强；虚线：待补齐的技能闭环与固定学习快照。",
        "status2": "当前技能主要是提示建议；完整闭环、预测校准及科研收益尚待验证。",
        "proposed": "待补齐",
    },
    "en": {
        "title": "FarField: Grounded Jumps and Branch-Scoped Evolution",
        "subtitle": "Proposed framework · Real observations inform future research and reusable capabilities",
        "tag": "PAPERFUTURE / FRAMEWORK v0.1",
        "goal": "Research intent + initial materials",
        "goal_flow": "Inspect → provisional questions, bottlenecks & unknowns",
        "goal_note": "Stable intent and constraints; revisable scientific subquestions",
        "a": "Mechanism-guided jumps", "a_sub": "GROUND · TRANSFER · INVESTIGATE",
        "b": "OpenWorld: learn through action", "b_sub": "PREDICT · ACT · OBSERVE · RESPOND",
        "c": "Branch-scoped evolution", "c_sub": "CONDITIONAL MEMORY & EXECUTABLE SKILLS",
        "a1": "Technical trajectory & bottleneck",
        "a1_body": ["Model drafts; sources check changes and limits"],
        "timeline": ["Prior methods", "Mechanism shift", "Bottleneck"],
        "a2": "A testable mechanism bridge",
        "a2_body": ["Source mechanism → shared structure → target", "State assumptions, alternatives and failure conditions"],
        "a2_note": "Underexplored, conditionally valuable, testable",
        "a3": "Landing inquiry & discriminating test",
        "a3_body": ["Check related methods, negative results and scope", "Design a cheap test; keep implementation open"],
        "a3_note": "Investigate further / revise the landing / defer",
        "a_footer": "Jump when needed; refine locally using real findings",
        "b1": "Expectations & discriminating predictions",
        "b1_body": ["Record alternatives, interventions and predictions", "Explore freely; freeze the protocol for confirmation"],
        "b2": "Build capabilities & execute",
        "b2_body": ["Write methods, access data, construct measurements", "Independent native evaluator + code and raw results"],
        "b2_note": "Real environment and observations ≠ internal rollout",
        "b3": "Validate measurements & respond",
        "b3_body": ["Check implementation, intervention and test validity", "Cite results: retain / revise / test further / pause"],
        "b3_note": "Prediction error alone does not refute a hypothesis",
        "performance": "Method performance", "mechanism": "Mechanism evidence",
        "separate": "Separate tests",
        "c1": "Conditional research memory",
        "c1_body": ["Findings, failure boundaries and unresolved questions", "Source observations, scope and evidence dependencies"],
        "c2": "Executable skill versions",
        "c2_body": ["Reusable analysis code and experimental procedures", "Local trial → validate → reuse / deactivate"],
        "c2_note": "Evolve research capabilities; keep shared rules fixed",
        "c3": "Fork a fixed snapshot; evolve separately",
        "branch_parent": "Parent · learning bundle vₜ",
        "branch_a": "Branch A · memory / skills",
        "branch_b": "Branch B · memory / skills",
        "c_footer": "Shared read-only sources; isolated learning updates",
        "candidate": "Idea", "feedback": "Results",
        "loop": "Evidence-linked memory and skills guide the next question, experiment or jump",
        "retraction": "Retracted evidence triggers dependency review; retaining a justified belief is allowed",
        "base_title": "Shared foundation",
        "base": "Literature & data  ·  Isolated execution  ·  Independent evaluation  ·  Event ledger & dependencies  ·  Budget",
        "outcome_title": "Research objective",
        "outcome": "Valuable directions, independently validated outcomes, and better subsequent research",
        "status_title": "Implementation boundary",
        "status1": "Solid: existing foundations requiring further work. Dashed: executable skill lifecycle and frozen learning snapshots to be completed.",
        "status2": "Skills currently remain prompt advice; the full loop, prediction calibration and research benefits are not yet established.",
        "proposed": "PLANNED",
    },
}


def draw(lang):
    d = COPY[lang]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    fontManager.addfont(CN_FONT)
    cn_family = FontProperties(fname=CN_FONT).get_name()
    families = ["DejaVu Sans", cn_family] if lang == "zh" else ["DejaVu Sans"]
    font = FontProperties(family=families)
    text_items = []

    def txt(x, y, s, size=22, color=None, weight="normal", align="left", maxw=None, bg=False):
        prop = font.copy()
        prop.set_weight(weight)
        t = ax.text(x, y, s, fontproperties=prop, fontsize=size * 0.72,
                    color=color or C["ink"], ha=align, va="center", zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 6} if bg else None)
        if maxw:
            text_items.append((t, maxw))
        return t

    def box(x, y, w, h, fill="white", edge=None, radius=12, dashed=False, lw=1.5):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                          linewidth=lw, facecolor=fill, edgecolor=edge or C["line"],
                          linestyle=(0, (6, 4)) if dashed else "solid", zorder=1)
        ax.add_patch(p)
        return p

    def arrow(x1, y1, x2, y2, color=None, lw=2, ms=13):
        p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                            linewidth=lw, color=color or C["muted"], shrinkA=0,
                            shrinkB=0, zorder=3)
        ax.add_patch(p)

    def line(x1, y1, x2, y2, color=None, lw=1.5):
        ax.plot([x1, x2], [y1, y2], color=color or C["line"], lw=lw, zorder=2)

    def card(x, y, title, body, accent, h, note=None, dashed=False, planned=False):
        box(x, y, 514, h, edge=accent if dashed else C["line"], dashed=dashed)
        title_w = 380 if planned else 472
        txt(x + 20, y + 27, title, size=25, weight="bold", maxw=title_w)
        if planned:
            box(x + 410, y + 13, 86, 25, fill=C["amber_bg"], edge="none", radius=5)
            txt(x + 453, y + 25.5, d["proposed"], size=14, color=accent, align="center")
        for j, s in enumerate(body):
            txt(x + 20, y + 64 + 29 * j, s, size=22, maxw=474)
        if note:
            txt(x + 20, y + h - 20, note, size=19, color=accent, maxw=474)

    # Header and research-intent rail.
    txt(48, 52, d["title"], size=40, weight="bold", maxw=1770)
    txt(48, 95, d["subtitle"], size=23, color=C["muted"], maxw=1320)
    txt(1866, 98, d["tag"], size=15, color=C["muted"], align="right")
    box(48, 130, 1818, 66, fill=C["rail"], edge="none", radius=9)
    txt(70, 162, d["goal"], size=25, weight="bold", maxw=460)
    line(547, 147, 547, 179, color="#B8C5D2")
    txt(572, 153, d["goal_flow"], size=24, maxw=1220)
    txt(572, 178, d["goal_note"], size=18, color=C["muted"], maxw=1220)
    arrow(331, 197, 331, 231, color=C["blue"], lw=2)

    # Three cooperating mechanisms. Solid borders denote implemented foundations,
    # not established scientific effectiveness or fully mature integration.
    xs = [48, 674, 1300]
    colors = [C["blue"], C["teal"], C["amber"]]
    for x, key, accent, fill, letter in zip(xs, ["a", "b", "c"], colors,
                                          [C["blue_bg"], C["teal_bg"], C["amber_bg"]], "ABC"):
        box(x, 234, 566, 600, fill=fill, edge=accent, radius=15, lw=1.4)
        ax.add_patch(Circle((x + 37, 267), 17, color=accent, zorder=2))
        # ASCII panel letter uses a Latin font for robust glyph rendering.
        ax.text(x + 37, 267, letter, fontsize=13, color="white", weight="bold",
                ha="center", va="center", fontfamily="DejaVu Sans", zorder=4)
        txt(x + 68, 266, d[key], size=28, color=accent, weight="bold", maxw=474)
        txt(x + 26, 296, d[key + "_sub"], size=16, color=accent, maxw=514)

    # A. Technical history -> mechanism transfer -> inquiry at the landing.
    xa = 74
    card(xa, 316, d["a1"], d["a1_body"], C["blue"], 120)
    for j, label in enumerate(d["timeline"]):
        px = xa + 19 + j * 164
        box(px, 403, 146, 22, fill=C["blue_bg"], edge="none", radius=4)
        txt(px + 73, 414, label, size=16, color=C["blue"], align="center", maxw=140)
        if j < 2:
            arrow(px + 149, 414, px + 161, 414, color=C["blue"], lw=1.1, ms=7)
    arrow(331, 440, 331, 454, color=C["blue"], lw=1.7, ms=9)
    card(xa, 458, d["a2"], d["a2_body"], C["blue"], 140, d["a2_note"])
    arrow(331, 602, 331, 616, color=C["blue"], lw=1.7, ms=9)
    card(xa, 620, d["a3"], d["a3_body"], C["blue"], 152, d["a3_note"])
    txt(331, 802, d["a_footer"], size=20, color=C["blue"], align="center", maxw=518)

    # B. The world-model expectation is distinct from the actual execution layer.
    xb = 700
    card(xb, 316, d["b1"], d["b1_body"], C["teal"], 120)
    arrow(957, 440, 957, 454, color=C["teal"], lw=1.7, ms=9)
    card(xb, 458, d["b2"], d["b2_body"], C["teal"], 140, d["b2_note"])
    arrow(957, 602, 957, 616, color=C["teal"], lw=1.7, ms=9)
    card(xb, 620, d["b3"], d["b3_body"], C["teal"], 152, d["b3_note"])
    for x, label in [(700, d["performance"]), (1029, d["mechanism"])]:
        box(x, 786, 185, 33, fill="white", edge="#A7CCC8", radius=6)
        txt(x + 92.5, 802.5, label, size=18, align="center", color=C["teal"], maxw=173)
    txt(957, 802.5, d["separate"], size=17, align="center", color=C["teal"], maxw=135)

    # C. Current lineage memory, plus clearly marked planned capabilities.
    xc = 1326
    card(xc, 316, d["c1"], d["c1_body"], C["amber"], 120)
    arrow(1583, 440, 1583, 454, color=C["amber"], lw=1.7, ms=9)
    card(xc, 458, d["c2"], d["c2_body"], C["amber"], 140,
         d["c2_note"], dashed=True, planned=True)
    arrow(1583, 602, 1583, 616, color=C["amber"], lw=1.7, ms=9)
    box(xc, 620, 514, 152, edge=C["amber"], dashed=True)
    txt(xc + 20, 646, d["c3"], size=23, weight="bold", maxw=474)
    box(xc + 117, 666, 280, 30, fill=C["amber_bg"], edge="none", radius=5)
    txt(xc + 257, 681, d["branch_parent"], size=18, align="center", maxw=270)
    line(1583, 697, 1583, 708, color=C["amber"])
    line(1448, 708, 1718, 708, color=C["amber"])
    for cx, label, fill, accent in [(1448, d["branch_a"], C["blue_bg"], C["blue"]),
                                   (1718, d["branch_b"], "#F5F1FA", C["purple"])]:
        arrow(cx, 708, cx, 722, color=C["amber"], lw=1.3, ms=8)
        box(cx - 115, 724, 230, 33, fill=fill, edge="none", radius=5)
        txt(cx, 740.5, label, size=17, color=accent, align="center", maxw=219)
    txt(1583, 802, d["c_footer"], size=20, color=C["amber"], align="center", maxw=520)

    # Inter-module arrows and the feedback loop.
    for start, end, label in [(622, 666, d["candidate"]), (1248, 1292, d["feedback"])]:
        arrow(start, 269, end, 269, color=C["muted"], lw=2, ms=12)
        txt((start + end) / 2, 248, label, size=15, color=C["muted"], align="center", maxw=56)
    vertices = [(1583, 835), (1583, 887), (1583, 900), (1570, 900),
                (344, 900), (331, 900), (331, 887), (331, 838)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO]
    feedback_path = MplPath(vertices, codes)
    ax.add_patch(PathPatch(feedback_path, fill=False, edgecolor=C["amber"], linewidth=2.2, zorder=2))
    arrow(331, 851, 331, 838, color=C["amber"], lw=2.2, ms=13)
    txt(957, 900, d["loop"], size=24, color=C["amber"], align="center", maxw=1160, bg=True)
    txt(957, 937, d["retraction"], size=20, color=C["muted"], align="center", maxw=1660)

    # Shared services and the objective sit beneath the research loop.
    box(48, 973, 1818, 50, fill=C["rail"], edge="none", radius=8)
    txt(68, 998, d["base_title"], size=21, weight="bold", maxw=235)
    line(317, 985, 317, 1011, color="#BBC6D2")
    txt(341, 998, d["base"], size=21, color=C["muted"], maxw=1494)
    txt(68, 1056, d["outcome_title"], size=22, weight="bold", maxw=235)
    txt(341, 1056, d["outcome"], size=23, maxw=1494)
    line(48, 1086, 1866, 1086)
    txt(48, 1111, d["status_title"], size=19, weight="bold", maxw=270)
    txt(330, 1111, d["status1"], size=18, color=C["muted"], maxw=1536)
    txt(330, 1142, d["status2"], size=18, color=C["muted"], maxw=1536)

    # Fit each line into its assigned region. Never silently clip text.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t, maxw in text_items:
        measured = t.get_window_extent(renderer=renderer).width
        if measured > maxw:
            t.set_fontsize(t.get_fontsize() * maxw / measured * 0.98)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t, maxw in text_items:
        if t.get_window_extent(renderer=renderer).width > maxw + 1:
            raise RuntimeError(f"Text overflow: {t.get_text()}")

    stem = OUT / f"farfield_framework_{lang}"
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig.savefig(stem.with_suffix(".svg"), format="svg", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), format="png", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"Rendered {stem.name}: SVG / PDF / PNG")


if __name__ == "__main__":
    for language in ("zh", "en"):
        draw(language)
