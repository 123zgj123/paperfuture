"""Illustrate idea formation through FarField's intended remote-jump mechanism.

This is a mechanism schematic, not a claim that the production retrieval
backend has achieved reliable mechanism grounding. Run with Python + matplotlib.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paperfuture-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
W, H = 2080, 1040
INK, MUTED, LINE = "#182D42", "#617285", "#D5DFE8"
BLUE, PURPLE, TEAL, AMBER = "#3569AF", "#7754AC", "#197F7A", "#A86A28"
PALE_BLUE, PALE_PURPLE, PALE_TEAL, PALE_AMBER = "#F0F5FC", "#F5F1FB", "#EFF8F6", "#FCF5EA"

WORDS = {
    "zh": {
        "title": "FarField 远跳机制：idea 如何形成",
        "sub": "沿技术演变定位瓶颈，以可迁移机制打开新的方法空间",
        "input": "输入：研究意图、已有方法与文献、上一跳的反馈（如有）",
        "trigger": "触发：首跳 / 路线停滞 / 主动探索新分支",
        "titles": ["定位跳跃起点", "寻找外部机制", "建立迁移桥梁", "落点研究与 idea 成形"],
        "subs": ["为什么需要新的方向？", "哪里有可能解决类似约束？", "什么关系可以迁移？", "具体提出什么新方法？"],
        "history": ["先前方法", "关键机制变化", "当前方法与边界"],
        "history_notes": ["原先依赖哪些条件", "解决了哪个问题", "仍然在哪些条件下失效"],
        "bottleneck": "当前瓶颈 B",
        "bottleneck_detail": "表示 / 信息 / 优化 / 测量 / 资源",
        "a_note": "模型绘制脉络；文献核验变化与边界",
        "search": "跨域检索 + 前沿模型知识线索",
        "query": "B",
        "candidates": ["候选机制 M₁", "优先调查：机制 M₂", "候选机制 M₃"],
        "candidate_notes": ["来源问题与适用条件", "可能作用于当前瓶颈", "来源问题与适用条件"],
        "search_note": "提取：作用机制、来源、假设与失效条件",
        "search_note2": "候选可以被修订、搁置或全部拒绝",
        "source": "来源领域：机制如何起作用",
        "source_nodes": ["输入 X", "机制 M", "结果 Y"],
        "target": "目标问题：如何重构为新方法",
        "target_nodes": ["目标 x", "迁移 M′", "预测 y"],
        "mapping": "提出对应关系；核验迁移条件",
        "keep": "保留：关键结构、因果或约束关系",
        "change": "改变：表示、算法结构或执行流程",
        "break": "检查：关键假设、破坏差异与竞争解释",
        "bridge_note": "把借来的机制转化为目标方法设计",
        "landing": "落点附近调查",
        "landing_body": ["相近方法 · 已有负面结果 · 适用边界", "按具体知识缺口继续检索、阅读与修订"],
        "synthesis": "可检验的新 idea",
        "idea": ["要突破的瓶颈 B + 迁移机制 M′", "方法实现：具体改变什么、如何改变", "成立条件：关键假设与竞争解释", "验证入口：可区分预测 + 廉价实验"],
        "output_note": "输出候选想法；有效性由后续研究检验",
        "feedback": "上一跳的真实发现 → 修订瓶颈与迁移条件 → 指导下一跳或局部深化",
        "selection": "投入依据：条件性价值、关键未知与验证成本；概念距离仅用于拓宽检索。",
        "boundary": "设计示意：当前检索仍可能只返回概念，机制证据与落点核验尚未可靠贯通。",
    },
    "en": {
        "title": "FarField Remote Jumps: How a New Idea Emerges",
        "sub": "Trace technical progress, identify a bottleneck, and transfer a mechanism into a new method",
        "input": "Input: research intent, existing methods and literature, and prior findings when available",
        "trigger": "Trigger: first idea / stagnation / branch diversification",
        "titles": ["Locate the departure", "Seek external mechanisms", "Build the transfer bridge", "Investigate & form an idea"],
        "subs": ["Why seek a new direction?", "What addresses a related constraint?", "Which relations can transfer?", "What new method is proposed?"],
        "history": ["Earlier methods", "Mechanism change", "Current method & limits"],
        "history_notes": ["Conditions previously assumed", "The problem this change addressed", "Conditions where limitations remain"],
        "bottleneck": "Current bottleneck B",
        "bottleneck_detail": "Representation / information / optimization / measurement / resources",
        "a_note": "Model sketches; sources check changes and limits",
        "search": "Cross-domain retrieval + model-supplied leads",
        "query": "B",
        "candidates": ["Candidate mechanism M₁", "Investigate mechanism M₂", "Candidate mechanism M₃"],
        "candidate_notes": ["Source problem and conditions", "May address the target bottleneck", "Source problem and conditions"],
        "search_note": "Extract mechanism, sources, assumptions and limits",
        "search_note2": "Revise, defer, or reject any or all candidates",
        "source": "Source domain: how the mechanism operates",
        "source_nodes": ["Input X", "Mechanism M", "Outcome Y"],
        "target": "Target problem: reconstruct a new method",
        "target_nodes": ["Target x", "Transfer M′", "Prediction y"],
        "mapping": "Propose correspondences; check transfer conditions",
        "keep": "Preserve: structural, causal or constraint relations",
        "change": "Change: representations, algorithms or procedures",
        "break": "Check: assumptions, breaking differences, alternatives",
        "bridge_note": "Turn a borrowed mechanism into a target design",
        "landing": "Investigate the landing neighborhood",
        "landing_body": ["Related methods · negative findings · scope", "Search, read and revise around specific knowledge gaps"],
        "synthesis": "A testable new idea",
        "idea": ["Target bottleneck B + transferred mechanism M′", "Implementation: what changes and how", "Conditions: assumptions and competing explanations", "Test: discriminating prediction + cheap experiment"],
        "output_note": "A proposed idea; effectiveness awaits investigation",
        "feedback": "Prior observations revise bottlenecks and transfer conditions, guiding the next jump or local refinement",
        "selection": "Allocate effort by conditional value, unresolved questions and test cost; distance only broadens retrieval.",
        "boundary": "Intended mechanism: current retrieval may return concepts only; mechanism grounding and landing verification remain incomplete.",
    },
}


def draw(lang):
    d = WORDS[lang]
    cn = "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"
    fontManager.addfont(cn)
    family = ["DejaVu Sans", FontProperties(fname=cn).get_name()] if lang == "zh" else ["DejaVu Sans"]
    font = FontProperties(family=family)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    labels = []

    def text(x, y, s, size=23, color=INK, bold=False, center=False, width=None, bg=False):
        f = font.copy()
        f.set_weight("bold" if bold else "normal")
        artist = ax.text(x, y, s, fontproperties=f, fontsize=size * .72, color=color,
                         va="center", ha="center" if center else "left", zorder=5,
                         bbox=dict(facecolor="white", edgecolor="none", pad=7) if bg else None)
        if width:
            labels.append((artist, width))

    def box(x, y, w, h, fill="white", edge=LINE, r=10, lw=1.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                   facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1))

    def arrow(x1, y1, x2, y2, color=MUTED, lw=1.8, curve=0, dashed=False, size=12):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                    connectionstyle=f"arc3,rad={curve}", color=color,
                                    linewidth=lw, mutation_scale=size, shrinkA=0, shrinkB=0,
                                    linestyle=(0, (4, 4)) if dashed else "solid", zorder=2))

    text(42, 57, d["title"], size=43, bold=True, width=1950)
    text(42, 107, d["sub"], size=25, color=MUTED, width=1920)
    box(42, 147, 1996, 66, fill="#F4F6F9", edge="none")
    text(62, 172, d["input"], size=23, width=1310)
    text(62, 197, d["trigger"], size=18, color=MUTED, width=1310)
    # Right aligned tag keeps the mechanism schematic distinct from a system diagram.
    ax.text(2018, 180, "MECHANISM SCHEMATIC", fontfamily="DejaVu Sans", fontsize=12,
            color=MUTED, ha="right", va="center")

    panel_x = [42, 543, 1044, 1545]
    accents = [BLUE, PURPLE, TEAL, AMBER]
    fills = [PALE_BLUE, PALE_PURPLE, PALE_TEAL, PALE_AMBER]
    for i, (x, accent, fill) in enumerate(zip(panel_x, accents, fills)):
        box(x, 255, 493, 543, fill=fill, edge=accent, r=15)
        ax.add_patch(Circle((x + 32, 282), 16, facecolor=accent, edgecolor="none", zorder=2))
        text(x + 32, 282, str(i + 1), size=20, color="white", bold=True, center=True)
        text(x + 59, 282, d["titles"][i], size=28, bold=True, color=accent, width=414)
        text(x + 20, 319, d["subs"][i], size=21, color=accent, width=453)
        if i < 3:
            # Short upper connectors avoid running through candidate or mapping graphics.
            arrow(x + 410, 240, x + 584, 240, color=MUTED, lw=1.7, size=10)
    arrow(287, 215, 287, 252, BLUE)

    # 1. A temporal chain motivates the departure; it is not a causal proof.
    for j, (title, note) in enumerate(zip(d["history"], d["history_notes"])):
        y = 354 + j * 99
        box(64, y, 449, 78, edge="#BFCFE4")
        text(85, y + 25, title, size=25, bold=True, width=405)
        text(85, y + 56, note, size=21, color=MUTED, width=405)
        arrow(288, y + 82, 288, y + 95, BLUE, size=9)
    box(64, 653, 449, 85, fill="white", edge=BLUE, lw=1.8)
    text(85, 679, d["bottleneck"], size=26, color=BLUE, bold=True, width=405)
    text(85, 712, d["bottleneck_detail"], size=20, width=405)
    text(288, 768, d["a_note"], size=20, color=BLUE, center=True, width=450)

    # 2. Candidate mechanisms widen the search; emphasis means inquiry priority.
    text(565, 371, d["search"], size=23, bold=True, width=448)
    ax.add_patch(Circle((590, 535), 27, facecolor="white", edgecolor=PURPLE, lw=2))
    text(590, 535, d["query"], size=27, color=PURPLE, bold=True, center=True)
    for j, (title, note) in enumerate(zip(d["candidates"], d["candidate_notes"])):
        y = 411 + j * 100
        active = j == 1
        color = PURPLE if active else MUTED
        arrow(619, 535, 674, y + 37, color=color, curve=.12 * (j - 1),
              dashed=not active, lw=2.1 if active else 1.2, size=10)
        box(678, y, 336, 76, fill="white", edge=PURPLE if active else LINE, lw=2 if active else 1.2)
        text(696, y + 26, title, size=24, color=color, bold=active, width=300)
        text(696, y + 55, note, size=19, color=MUTED, width=300)
    text(565, 724, d["search_note"], size=20, color=PURPLE, width=448)
    text(565, 764, d["search_note2"], size=20, color=MUTED, width=448)

    # 3. Mapping shared relations is the center of the jump, not domain distance.
    text(1066, 366, d["source"], size=23, bold=True, width=448)
    source_centers = [1132, 1290, 1448]
    for j, (cx, name) in enumerate(zip(source_centers, d["source_nodes"])):
        box(cx - 66, 397, 132, 57, fill="white", edge=PURPLE)
        text(cx, 425, name, size=22, color=PURPLE, center=True, width=121)
        if j < 2:
            arrow(cx + 68, 425, source_centers[j + 1] - 68, 425, PURPLE, size=9)
        arrow(cx, 458, cx, 539, TEAL, lw=1.2, dashed=True, size=9)
    text(1290, 497, d["mapping"], size=20, color=TEAL, center=True, width=448, bg=True)
    for j, (cx, name) in enumerate(zip(source_centers, d["target_nodes"])):
        box(cx - 66, 544, 132, 57, fill="white", edge=TEAL)
        text(cx, 572, name, size=22, color=TEAL, center=True, width=121)
        if j < 2:
            arrow(cx + 68, 572, source_centers[j + 1] - 68, 572, TEAL, size=9)
    text(1066, 627, d["target"], size=22, bold=True, width=448)
    for j, key in enumerate(["keep", "change", "break"]):
        text(1066, 671 + j * 32, d[key], size=21, width=448)
    text(1290, 772, d["bridge_note"], size=20, color=TEAL, center=True, width=448)

    # 4. Landing investigation revises the proposed transfer into an explicit idea.
    box(1567, 354, 449, 117, edge="#D5C1A5")
    text(1587, 380, d["landing"], size=25, bold=True, width=409)
    for j, s in enumerate(d["landing_body"]):
        text(1587, 417 + j * 29, s, size=20, width=409)
    arrow(1792, 476, 1792, 504, AMBER, size=10)
    box(1567, 510, 449, 226, fill="white", edge=AMBER, lw=2)
    text(1587, 544, d["synthesis"], size=30, bold=True, color=AMBER, width=409)
    ax.plot([1587, 1996], [568, 568], color="#E8D7BC", lw=1)
    for j, s in enumerate(d["idea"]):
        ax.add_patch(Circle((1592, 597 + j * 35), 3.2, color=AMBER))
        text(1605, 597 + j * 35, s, size=21, width=389)
    text(1792, 768, d["output_note"], size=20, color=AMBER, center=True, width=449)

    # Only the informational feedback to future jumps is shown, not full-system modules.
    ax.plot([1792, 1792, 288, 288], [802, 883, 883, 815], color=PURPLE, lw=2, zorder=2)
    arrow(288, 817, 288, 802, PURPLE, lw=2)
    text(1040, 883, d["feedback"], size=25, color=PURPLE, center=True, width=1400, bg=True)
    text(1040, 944, d["selection"], size=23, center=True, width=1960)
    ax.plot([42, 2038], [977, 977], color=LINE, lw=1)
    text(42, 1007, d["boundary"], size=20, color=MUTED, width=1996)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t, width in labels:
        actual = t.get_window_extent(renderer).width
        if actual > width:
            t.set_fontsize(t.get_fontsize() * width / actual * .98)
    fig.canvas.draw()
    for t, width in labels:
        assert t.get_window_extent(fig.canvas.get_renderer()).width < width + 1, t.get_text()
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    stem = OUT / f"farfield_remote_jump_{lang}"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Rendered {stem.name}: PNG, SVG, PDF")


if __name__ == "__main__":
    for language in ("zh", "en"):
        draw(language)
