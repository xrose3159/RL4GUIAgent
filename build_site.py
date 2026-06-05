#!/usr/bin/env python3
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "gui_agent_rl_survey.md"


@dataclass
class Paper:
    number: int
    title: str
    link_text: str
    links: list[str]
    motivation: str
    method: str
    experiment: str
    tags: list[str]


CATEGORY_RULES = [
    ("Online RL", ["DigiRL", "WebRL", "MobileRL", "WebAgent-R1", "DART-GUI", "ComputerRL", "ZeroGUI", "AgentCPM"]),
    ("Offline RL", ["Digi-Q", "Agent Q", "UI-R1", "GUI-R1", "UI-TARS", "ARPO", "AEBPO"]),
    ("Grounding", ["GUI-G2", "SE-GUI", "InfiGUI-G1", "UI-AGILE", "GUI-Eyes", "InfiGUI-R1"]),
    ("Hybrid", ["UI-S1", "Hi-Agent", "UltraCUA", "Co-EPG", "Mano", "ClawGUI", "BacktrackAgent", "VSC-RL"]),
    ("World Model", ["DynaWeb", "WebSynthesis", "WebWorld", "Code2World"]),
    ("Data", ["Explorer", "AgentTrek", "GELab", "Step-GUI", "MAI-UI", "NestBrowse"]),
    ("Reward", ["ProgRM", "ZeroGUI", "GUI-G2", "Step-GUI"]),
    ("OS/Desktop", ["OS-Copilot", "ComputerRL", "UltraCUA", "DART-GUI"]),
    ("Mobile", ["DigiRL", "Digi-Q", "MobileRL", "Hi-Agent", "GELab", "MAI-UI"]),
    ("Web", ["WebRL", "Agent Q", "WebAgent-R1", "DynaWeb", "WebSynthesis", "WebWorld", "Explorer", "NestBrowse", "AgentTrek"]),
]


def inline_md(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<a href="\2" target="_blank" rel="noreferrer">\1</a>',
        text,
    )
    text = re.sub(
        r"(?<![\"'=])(https?://[^\s，。；、)]+)",
        r'<a href="\1" target="_blank" rel="noreferrer">\1</a>',
        text,
    )
    return text


def block_md(lines: list[str]) -> str:
    out: list[str] = []
    buf: list[str] = []

    def flush_paragraph() -> None:
        if buf:
            out.append(f"<p>{inline_md(' '.join(buf))}</p>")
            buf.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_paragraph()
            continue
        buf.append(line)
    flush_paragraph()

    # Second pass for simple unordered lists, preserving paragraphs above for non-list sections.
    if any(line.strip().startswith("- ") or re.match(r"^\d+\.\s+", line.strip()) for line in lines):
        out = []
        list_type: str | None = None
        para: list[str] = []

        def close_list() -> None:
            nonlocal list_type
            if list_type:
                out.append(f"</{list_type}>")
                list_type = None

        for raw in lines:
            line = raw.strip()
            if not line:
                if para:
                    out.append(f"<p>{inline_md(' '.join(para))}</p>")
                    para.clear()
                close_list()
                continue
            if line.startswith("- "):
                if para:
                    out.append(f"<p>{inline_md(' '.join(para))}</p>")
                    para.clear()
                if list_type != "ul":
                    close_list()
                    out.append("<ul>")
                    list_type = "ul"
                out.append(f"<li>{inline_md(line[2:])}</li>")
            elif re.match(r"^\d+\.\s+", line):
                if para:
                    out.append(f"<p>{inline_md(' '.join(para))}</p>")
                    para.clear()
                if list_type != "ol":
                    close_list()
                    out.append("<ol>")
                    list_type = "ol"
                item_text = re.sub(r"^\d+\.\s+", "", line)
                out.append(f"<li>{inline_md(item_text)}</li>")
            else:
                close_list()
                para.append(line)
        if para:
            out.append(f"<p>{inline_md(' '.join(para))}</p>")
        close_list()
    return "\n".join(out)


def extract_between(text: str, start: str, end: str | None = None) -> str:
    start_idx = text.index(start) + len(start)
    end_idx = text.index(end, start_idx) if end else len(text)
    return text[start_idx:end_idx].strip()


def tags_for(title: str) -> list[str]:
    tags: list[str] = []
    for tag, needles in CATEGORY_RULES:
        if any(needle.lower() in title.lower() for needle in needles):
            tags.append(tag)
    return tags or ["GUI RL"]


def parse_papers(section: str) -> list[Paper]:
    chunks = re.split(r"(?=^###\s+\d+\.\s+)", section, flags=re.M)
    papers: list[Paper] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^###\s+(\d+)\.\s+(.+)$", chunk, flags=re.M)
        if not m:
            continue
        number = int(m.group(1))
        title = m.group(2).strip()
        lines = chunk.splitlines()[1:]
        body = "\n".join(lines).strip()
        link_match = re.search(r"^链接：(.+)$", body, flags=re.M)
        link_text = link_match.group(1).strip() if link_match else ""
        links = re.findall(r"https?://[^\s，。；、]+", link_text)
        fields: dict[str, str] = {}
        for label, next_label in [("动机", "方案"), ("方案", "实验"), ("实验", None)]:
            pattern = rf"{label}：(.*?)(?=\n\n{next_label}：|\Z)" if next_label else rf"{label}：(.*)"
            fm = re.search(pattern, body, flags=re.S)
            fields[label] = re.sub(r"\s+", " ", fm.group(1)).strip() if fm else ""
        papers.append(
            Paper(
                number=number,
                title=title,
                link_text=link_text,
                links=links,
                motivation=fields["动机"],
                method=fields["方案"],
                experiment=fields["实验"],
                tags=tags_for(title),
            )
        )
    return papers


def parse_heading_sections(section: str) -> list[tuple[str, str]]:
    chunks = re.split(r"(?=^###\s+)", section, flags=re.M)
    parsed: list[tuple[str, str]] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        title = re.sub(r"^###\s+", "", lines[0]).strip()
        parsed.append((title, block_md(lines[1:])))
    return parsed


def parse_references(section: str) -> list[tuple[str, str]]:
    refs = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        item = line[2:]
        match = re.match(r"^(.+):\s+(https?://\S+)$", item)
        if match:
            name, url = match.groups()
            refs.append((name, url))
    return refs


def paper_card(paper: Paper) -> str:
    link_buttons = "".join(
        f'<a class="paper-link" href="{html.escape(url)}" target="_blank" rel="noreferrer">论文链接</a>'
        for url in paper.links
    )
    if not link_buttons and paper.link_text:
        link_buttons = f'<span class="paper-note">{inline_md(paper.link_text)}</span>'
    tags = "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in paper.tags)
    searchable = html.escape((paper.title + " " + " ".join(paper.tags)).lower())
    primary = html.escape(paper.tags[0])
    return f"""
    <article class="paper-card" id="paper-{paper.number}" data-tags="{' '.join(html.escape(t) for t in paper.tags)}" data-primary="{primary}" data-search="{searchable}">
      <div class="paper-top">
        <a class="paper-no" href="#paper-{paper.number}">{paper.number:02d}</a>
        <div>
          <h3>{inline_md(paper.title)}</h3>
          <div class="tag-row">{tags}</div>
        </div>
        <div class="paper-actions">{link_buttons}</div>
      </div>
      <div class="paper-grid">
        <section><h4>动机</h4><p>{inline_md(paper.motivation)}</p></section>
        <section><h4>方案</h4><p>{inline_md(paper.method)}</p></section>
        <section><h4>实验</h4><p>{inline_md(paper.experiment)}</p></section>
      </div>
    </article>
    """


def build() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    overview = extract_between(source, "## 0. 总体判断", "## 1. 核心方法论文")
    paper_section = extract_between(source, "## 1. 核心方法论文", "## 2. 训练环境、容器与技术栈")
    env_section = extract_between(source, "## 2. 训练环境、容器与技术栈", "## 3. 现在大家实际怎么做 GUI Agent RL")
    practice = extract_between(source, "## 3. 现在大家实际怎么做 GUI Agent RL", "## 4. 关键趋势")
    trends = extract_between(source, "## 4. 关键趋势", "## 5. 参考链接")
    refs = parse_references(extract_between(source, "## 5. 参考链接"))
    papers = parse_papers(paper_section)
    envs = parse_heading_sections(env_section)

    papers_html = "\n".join(paper_card(p) for p in papers)
    env_html = "\n".join(
        f'<article class="env-panel" id="env-{idx + 1}"><h3>{inline_md(title)}</h3>{body}</article>'
        for idx, (title, body) in enumerate(envs)
    )
    refs_html = "\n".join(
        f'<a href="{html.escape(url)}" target="_blank" rel="noreferrer">{html.escape(name)}</a>' for name, url in refs
    )

    overview_lines = [line.strip() for line in overview.splitlines() if line.strip()]
    thesis = overview_lines[0] if overview_lines else ""
    route_lines = [line for line in overview_lines if re.match(r"\d+\.", line)]
    routes_html = "\n".join(
        f'<div class="route"><strong>{inline_md(line.split("：", 1)[0])}</strong><span>{inline_md(line.split("：", 1)[1] if "：" in line else line)}</span></div>'
        for line in route_lines[:3]
    )

    practice_html = block_md(practice.splitlines())
    trends_html = block_md(trends.splitlines())

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GUI Agent RL 论文地图</title>
  <meta name="description" content="GUI Agent with Reinforcement Learning 论文梳理、环境技术栈与训练路线。">
  <link rel="icon" href="favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <a class="skip-link" href="#papers">跳到论文列表</a>
  <header class="site-header">
    <nav class="nav-shell" aria-label="站点导航">
      <a class="brand" href="#top">GUI Agent RL</a>
      <div class="nav-links">
        <a href="#map">路线</a>
        <a href="#papers">论文</a>
        <a href="#env">环境</a>
        <a href="#practice">做法</a>
        <a href="#refs">参考</a>
      </div>
    </nav>
  </header>

  <main id="top">
    <section class="hero" aria-labelledby="hero-title">
      <div class="hero-copy">
        <p class="eyebrow">arXiv:2604.27955 阅读笔记</p>
        <h1 id="hero-title">GUI Agent RL 论文地图</h1>
        <p class="lead">{inline_md(thesis)}</p>
        <div class="hero-actions">
          <a class="primary-action" href="#papers">浏览 41 篇论文</a>
          <a class="secondary-action" href="https://arxiv.org/abs/2604.27955" target="_blank" rel="noreferrer">打开综述原文</a>
        </div>
      </div>
      <aside class="signal-panel" aria-label="报告范围">
        <dl>
          <div><dt>论文</dt><dd>{len(papers)}</dd></div>
          <div><dt>路线</dt><dd>3</dd></div>
          <div><dt>环境</dt><dd>Web / OS / Mobile</dd></div>
        </dl>
      </aside>
    </section>

    <section class="section-block" id="map" aria-labelledby="map-title">
      <div class="section-heading">
        <p class="eyebrow">Taxonomy</p>
        <h2 id="map-title">三条主线</h2>
      </div>
      <div class="route-grid">{routes_html}</div>
    </section>

    <section class="section-block" id="papers" aria-labelledby="papers-title">
      <div class="section-heading wide">
        <div>
          <p class="eyebrow">Paper Notes</p>
          <h2 id="papers-title">逐篇论文细读</h2>
        </div>
        <div class="toolbar" role="search">
          <input id="paper-search" type="search" placeholder="搜索论文、路线或环境" aria-label="搜索论文">
          <select id="paper-filter" aria-label="按类别筛选">
            <option value="all">全部类别</option>
            <option>Online RL</option>
            <option>Offline RL</option>
            <option>Grounding</option>
            <option>Hybrid</option>
            <option>World Model</option>
            <option>Reward</option>
            <option>Web</option>
            <option>Mobile</option>
            <option>OS/Desktop</option>
          </select>
        </div>
      </div>
      <p class="result-line"><span id="paper-count">{len(papers)}</span> 篇论文显示中</p>
      <div class="paper-list">{papers_html}</div>
    </section>

    <section class="section-block" id="env" aria-labelledby="env-title">
      <div class="section-heading">
        <p class="eyebrow">Environments</p>
        <h2 id="env-title">环境、容器与框架</h2>
      </div>
      <div class="env-grid">{env_html}</div>
    </section>

    <section class="section-block split" id="practice" aria-labelledby="practice-title">
      <div class="section-heading sticky-heading">
        <p class="eyebrow">Recipe</p>
        <h2 id="practice-title">现在大家实际怎么做</h2>
      </div>
      <div class="prose">{practice_html}</div>
    </section>

    <section class="section-block split" id="trends" aria-labelledby="trends-title">
      <div class="section-heading sticky-heading">
        <p class="eyebrow">Trends</p>
        <h2 id="trends-title">关键趋势</h2>
      </div>
      <div class="prose">{trends_html}</div>
    </section>

    <section class="section-block" id="refs" aria-labelledby="refs-title">
      <div class="section-heading">
        <p class="eyebrow">Sources</p>
        <h2 id="refs-title">参考链接</h2>
      </div>
      <div class="refs-grid">{refs_html}</div>
    </section>
  </main>

  <footer class="site-footer">
    <p>静态站点，由 Markdown 报告自动生成，可直接部署到 GitHub Pages。</p>
  </footer>

  <script src="script.js"></script>
</body>
</html>
"""

    (ROOT / "index.html").write_text(html_doc, encoding="utf-8")
    (ROOT / "styles.css").write_text(CSS, encoding="utf-8")
    (ROOT / "script.js").write_text(JS, encoding="utf-8")
    (ROOT / "favicon.svg").write_text(FAVICON, encoding="utf-8")
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    (ROOT / "README.md").write_text(README, encoding="utf-8")


CSS = r"""
:root {
  color-scheme: light;
  --paper: oklch(96.8% 0.018 78);
  --paper-strong: oklch(91.5% 0.024 78);
  --ink: oklch(21% 0.027 255);
  --muted: oklch(43% 0.035 252);
  --rule: oklch(77% 0.035 75);
  --red: oklch(55% 0.18 27);
  --blue: oklch(41% 0.12 236);
  --green: oklch(47% 0.10 155);
  --amber: oklch(70% 0.14 82);
  --shadow: 0 20px 60px color-mix(in oklch, var(--ink) 15%, transparent);
  --radius: 8px;
  --max: 1180px;
  --sans: ui-sans-serif, "Avenir Next", "PingFang SC", "Microsoft YaHei", sans-serif;
  --serif: Georgia, "Songti SC", "Noto Serif CJK SC", serif;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  background:
    linear-gradient(90deg, color-mix(in oklch, var(--rule) 18%, transparent) 1px, transparent 1px) 0 0 / 44px 44px,
    linear-gradient(0deg, color-mix(in oklch, var(--rule) 16%, transparent) 1px, transparent 1px) 0 0 / 44px 44px,
    var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  line-height: 1.72;
  letter-spacing: 0;
}

a {
  color: var(--blue);
  text-decoration-thickness: 0.08em;
  text-underline-offset: 0.18em;
}

.skip-link {
  position: absolute;
  left: 1rem;
  top: -4rem;
  z-index: 100;
  background: var(--ink);
  color: var(--paper);
  padding: 0.55rem 0.8rem;
  border-radius: var(--radius);
}

.skip-link:focus {
  top: 1rem;
}

.site-header {
  position: sticky;
  top: 0;
  z-index: 20;
  border-bottom: 1px solid color-mix(in oklch, var(--rule) 70%, transparent);
  background: color-mix(in oklch, var(--paper) 88%, transparent);
  backdrop-filter: blur(14px);
}

.nav-shell {
  width: min(var(--max), calc(100% - 32px));
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 64px;
  gap: 1rem;
}

.brand {
  color: var(--ink);
  font-weight: 800;
  text-decoration: none;
  letter-spacing: 0;
}

.nav-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  justify-content: flex-end;
}

.nav-links a {
  color: var(--muted);
  text-decoration: none;
  padding: 0.35rem 0.62rem;
  border-radius: var(--radius);
  font-size: 0.92rem;
}

.nav-links a:hover,
.nav-links a:focus-visible {
  background: var(--paper-strong);
  color: var(--ink);
}

main {
  width: min(var(--max), calc(100% - 32px));
  margin: 0 auto;
}

.hero {
  min-height: calc(100svh - 64px);
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(260px, 360px);
  align-items: center;
  gap: clamp(2rem, 6vw, 5rem);
  padding: clamp(3rem, 8vw, 7rem) 0 clamp(2rem, 5vw, 4rem);
  border-bottom: 2px solid var(--ink);
}

.eyebrow {
  margin: 0 0 0.65rem;
  color: var(--red);
  font-size: 0.78rem;
  font-weight: 800;
  text-transform: uppercase;
}

h1,
h2,
h3,
h4 {
  margin: 0;
  line-height: 1.15;
  letter-spacing: 0;
}

h1 {
  max-width: 850px;
  font-family: var(--serif);
  font-size: clamp(3.2rem, 8vw, 7.4rem);
  font-weight: 700;
}

.lead {
  max-width: 760px;
  margin: 1.5rem 0 0;
  color: color-mix(in oklch, var(--ink) 82%, var(--muted));
  font-size: clamp(1rem, 1.8vw, 1.22rem);
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 2rem;
}

.primary-action,
.secondary-action,
.paper-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 40px;
  padding: 0.55rem 0.9rem;
  border: 1px solid var(--ink);
  border-radius: var(--radius);
  text-decoration: none;
  font-weight: 750;
}

.primary-action {
  background: var(--ink);
  color: var(--paper);
}

.secondary-action,
.paper-link {
  color: var(--ink);
  background: color-mix(in oklch, var(--paper) 84%, white);
}

.signal-panel {
  align-self: stretch;
  display: grid;
  align-content: end;
}

.signal-panel dl {
  display: grid;
  gap: 1px;
  margin: 0;
  border: 2px solid var(--ink);
  background: var(--ink);
  box-shadow: var(--shadow);
}

.signal-panel div {
  padding: 1.1rem;
  background: var(--paper-strong);
}

.signal-panel dt {
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 750;
}

.signal-panel dd {
  margin: 0.18rem 0 0;
  font-family: var(--serif);
  font-size: clamp(1.4rem, 3vw, 2.4rem);
  font-weight: 700;
}

.section-block {
  padding: clamp(3rem, 7vw, 5.5rem) 0;
  border-bottom: 1px solid color-mix(in oklch, var(--rule) 80%, transparent);
}

.section-heading {
  max-width: 760px;
  margin-bottom: clamp(1.5rem, 4vw, 2.7rem);
}

.section-heading.wide {
  max-width: none;
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 1rem;
}

h2 {
  font-family: var(--serif);
  font-size: clamp(2.1rem, 5vw, 4rem);
}

.route-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
}

.route {
  min-height: 220px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 1.5rem;
  padding: 1.1rem;
  border-top: 4px solid var(--ink);
  background: color-mix(in oklch, var(--paper) 92%, white);
}

.route strong {
  font-size: clamp(1.1rem, 2vw, 1.45rem);
}

.route span {
  color: var(--muted);
}

.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  justify-content: flex-end;
}

input,
select {
  min-height: 42px;
  border: 1px solid var(--ink);
  border-radius: var(--radius);
  background: color-mix(in oklch, var(--paper) 88%, white);
  color: var(--ink);
  font: inherit;
  padding: 0.45rem 0.7rem;
}

input {
  width: min(320px, 100%);
}

.result-line {
  margin: -1.2rem 0 1.1rem;
  color: var(--muted);
  font-size: 0.94rem;
}

.paper-list {
  display: grid;
  gap: 1rem;
}

.paper-card {
  background: color-mix(in oklch, var(--paper) 90%, white);
  border: 1px solid color-mix(in oklch, var(--ink) 24%, var(--rule));
  border-radius: var(--radius);
  overflow: hidden;
}

.paper-card[hidden] {
  display: none;
}

.paper-top {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 1rem;
  padding: 1rem;
  border-bottom: 1px solid color-mix(in oklch, var(--rule) 75%, transparent);
}

.paper-no {
  color: var(--red);
  font-family: var(--serif);
  font-size: 1.5rem;
  font-weight: 700;
  text-decoration: none;
}

.paper-top h3 {
  font-size: clamp(1.05rem, 2vw, 1.35rem);
}

.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.55rem;
}

.tag {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 0.08rem 0.45rem;
  border: 1px solid color-mix(in oklch, var(--ink) 35%, var(--rule));
  border-radius: 999px;
  color: color-mix(in oklch, var(--muted) 80%, var(--ink));
  font-size: 0.74rem;
  font-weight: 700;
}

.paper-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  justify-content: flex-end;
}

.paper-link {
  min-height: 32px;
  padding: 0.25rem 0.55rem;
  font-size: 0.8rem;
}

.paper-note {
  max-width: 240px;
  color: var(--muted);
  font-size: 0.86rem;
}

.paper-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.paper-grid section {
  padding: 1rem;
  border-right: 1px solid color-mix(in oklch, var(--rule) 70%, transparent);
}

.paper-grid section:last-child {
  border-right: 0;
}

.paper-grid h4 {
  color: var(--red);
  font-size: 0.82rem;
  margin-bottom: 0.45rem;
}

.paper-grid p,
.prose p,
.env-panel p,
.env-panel li,
.prose li {
  margin: 0;
  color: color-mix(in oklch, var(--ink) 82%, var(--muted));
  font-size: 0.96rem;
}

.env-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

.env-panel {
  padding: 1rem;
  border-left: 4px solid var(--green);
  background: color-mix(in oklch, var(--paper) 90%, white);
}

.env-panel h3 {
  margin-bottom: 0.75rem;
  font-size: 1.25rem;
}

.env-panel p + p,
.prose p + p,
.prose ul + p,
.env-panel ul + p {
  margin-top: 0.8rem;
}

.env-panel ul,
.prose ul {
  margin: 0.55rem 0 0;
  padding-left: 1.15rem;
}

.split {
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr);
  gap: clamp(1.5rem, 5vw, 4rem);
}

.sticky-heading {
  position: sticky;
  top: 88px;
  align-self: start;
}

.prose {
  max-width: 760px;
}

.refs-grid {
  columns: 3 220px;
  column-gap: 1rem;
}

.refs-grid a {
  display: block;
  break-inside: avoid;
  margin: 0 0 0.42rem;
  color: var(--blue);
}

.site-footer {
  width: min(var(--max), calc(100% - 32px));
  margin: 0 auto;
  padding: 2rem 0 3rem;
  color: var(--muted);
  font-size: 0.92rem;
}

code {
  background: color-mix(in oklch, var(--amber) 28%, transparent);
  padding: 0.05rem 0.24rem;
  border-radius: 4px;
}

@media (max-width: 860px) {
  .hero,
  .route-grid,
  .paper-grid,
  .env-grid,
  .split {
    grid-template-columns: 1fr;
  }

  .hero {
    min-height: auto;
  }

  .section-heading.wide,
  .paper-top {
    display: block;
  }

  .toolbar,
  .paper-actions {
    justify-content: flex-start;
    margin-top: 0.85rem;
  }

  input {
    width: 100%;
  }

  .paper-grid section {
    border-right: 0;
    border-bottom: 1px solid color-mix(in oklch, var(--rule) 70%, transparent);
  }

  .paper-grid section:last-child {
    border-bottom: 0;
  }

  .sticky-heading {
    position: static;
  }
}

@media (max-width: 560px) {
  .nav-shell {
    align-items: flex-start;
    flex-direction: column;
    padding: 0.75rem 0;
  }

  .nav-links {
    justify-content: flex-start;
  }

  h1 {
    font-size: clamp(2.7rem, 18vw, 4.1rem);
  }
}
"""

JS = r"""
const search = document.querySelector("#paper-search");
const filter = document.querySelector("#paper-filter");
const cards = [...document.querySelectorAll(".paper-card")];
const count = document.querySelector("#paper-count");

function applyFilters() {
  const query = (search.value || "").trim().toLowerCase();
  const selected = filter.value;
  let visible = 0;

  for (const card of cards) {
    const text = card.dataset.search || "";
    const tags = card.dataset.tags || "";
    const matchesQuery = !query || text.includes(query);
    const matchesTag = selected === "all" || tags.includes(selected);
    const show = matchesQuery && matchesTag;
    card.hidden = !show;
    if (show) visible += 1;
  }

  count.textContent = String(visible);
}

search?.addEventListener("input", applyFilters);
filter?.addEventListener("change", applyFilters);
applyFilters();

document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", () => {
    const target = document.querySelector(anchor.getAttribute("href"));
    if (!target) return;
    target.setAttribute("tabindex", "-1");
    setTimeout(() => target.focus({ preventScroll: true }), 350);
  });
});
"""

FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="8" fill="#f3ecdf"/>
  <path d="M12 14h40v34H12z" fill="none" stroke="#111827" stroke-width="4"/>
  <path d="M18 24h28M18 33h20M18 42h14" stroke="#b33b2e" stroke-width="4" stroke-linecap="square"/>
</svg>
"""

README = """# GUI Agent RL 论文地图

这是一个可直接部署到 GitHub Pages 的静态 HTML 站点。内容由上一级目录的 `gui_agent_rl_survey.md` 自动生成。

## 本地预览

```bash
python3 -m http.server 8000
```

然后打开 <http://localhost:8000>。

## 重新生成

```bash
python3 build_site.py
```

## 发布到 GitHub Pages

当前机器需要先登录 GitHub CLI：

```bash
gh auth login
```

登录后可在本目录执行：

```bash
git init
git add .
git commit -m "Publish GUI Agent RL survey site"
gh repo create gui-agent-rl-survey --public --source=. --remote=origin --push
gh api -X POST "repos/$(gh api user -q .login)/gui-agent-rl-survey/pages" -f source[branch]=main -f source[path]=/
```

发布成功后，站点地址通常是：

```text
https://<你的 GitHub 用户名>.github.io/gui-agent-rl-survey/
```
"""


if __name__ == "__main__":
    build()
