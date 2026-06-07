import {
  deriveSignals,
  extractExperimentFacts,
  loadManifest,
  loadPaperDocument,
  mdSnippet,
  siteUrl,
} from "./paper-data.js";
import { hydrateMarkdown, renderMarkdown, waitForMarkdownRuntime } from "./markdown-renderer.js";

function escapeHtml(value = "") {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function linkList(links = []) {
  if (!links.length) return "";
  return links
    .map((link, index) => {
      const klass = index === 0 ? "primary-action" : "secondary-action";
      return `<a class="${klass}" href="${escapeHtml(link.url)}" target="_blank" rel="noreferrer">${escapeHtml(link.label || "论文链接")}</a>`;
    })
    .join("");
}

function chipList(values = []) {
  return values.map((value) => `<span class="tag">${escapeHtml(value)}</span>`).join("");
}

function localDetailHref(path = "") {
  return path.replace(/^papers\//, "");
}

function experimentBoard(facts) {
  if (!facts.length) return "";
  return `<div class="experiment-board" aria-label="从 Markdown 第 2 部分抽取的实验数据看板">
    ${facts
      .map(
        (fact) => `<article>
          <span>${escapeHtml(fact.label)}</span>
          <p>${escapeHtml(fact.text)}</p>
        </article>`,
      )
      .join("")}
  </div>`;
}

class LiteratureDetail extends HTMLElement {
  async connectedCallback() {
    this.paperId = this.getAttribute("paper-id");
    this.renderLoading();
    try {
      const manifest = await loadManifest();
      const papers = manifest.papers || [];
      const paper = papers.find((item) => item.id === this.paperId) || papers[0];
      if (!paper) throw new Error("No paper metadata found.");
      const loaded = await loadPaperDocument(paper);
      await waitForMarkdownRuntime();
      this.renderPaper(loaded.paper, loaded.document, papers);
    } catch (error) {
      this.renderError(error);
    }
  }

  renderLoading() {
    this.innerHTML = `<section class="reader-state"><p class="eyebrow">Loading Markdown</p><h1>正在读取文献数据库...</h1></section>`;
  }

  renderError(error) {
    this.innerHTML = `<section class="reader-state"><p class="eyebrow">Load failed</p><h1>文献加载失败</h1><p>${escapeHtml(error.message)}</p></section>`;
  }

  renderPaper(paper, document, papers) {
    const signals = deriveSignals(paper, document);
    const methods = document.sections.methods?.content || "";
    const experiments = document.sections.experiments?.content || "";
    const visualGuides = document.sections.visualGuides?.content || "";
    const limitations = document.sections.limitations?.content || "";
    const facts = extractExperimentFacts(experiments);
    const figure = paper.figurePath
      ? `<figure class="reader-figure"><a href="${siteUrl(paper.figurePath)}" target="_blank" rel="noreferrer"><img src="${siteUrl(paper.figurePath)}" alt="${escapeHtml(paper.title)} framework figure" loading="lazy"></a><figcaption>论文框架图/关键图。正文的 Visual Guides 仍然来自 Markdown 第 3 部分。</figcaption></figure>`
      : "";
    const index = papers.findIndex((item) => item.id === paper.id);
    const prev = papers[index - 1];
    const next = papers[index + 1];

    this.innerHTML = `
      <article class="reader-shell">
        <aside class="reader-rail">
          <a class="reader-back" href="../index.html#papers">返回论文库</a>
          <div class="reader-source">
            <span>Source</span>
            <code>${escapeHtml(paper.docPath)}</code>
          </div>
          <nav class="toc-panel" aria-label="文献详情目录">
            <a href="#methods">方法深度解析</a>
            <a href="#experiments">实验数据看板</a>
            <a href="#visual-guides">图文占位符</a>
            <a href="#limitations">局限方向</a>
          </nav>
          <div class="reader-meta">
            <span>Method</span>
            <strong>${escapeHtml(signals.methodFamily)}</strong>
          </div>
          <div class="reader-meta">
            <span>Environment</span>
            <strong>${escapeHtml(signals.environment.join(" / "))}</strong>
          </div>
        </aside>

        <div class="reader-main">
          <header class="reader-hero">
            <p class="eyebrow">Paper ${escapeHtml(paper.id)} · Markdown-driven Detail</p>
            <h1>${escapeHtml(document.title || paper.fullTitle || paper.title)}</h1>
            <div class="tag-row">${chipList([...paper.tags, signals.methodFamily, ...signals.environment])}</div>
            <p class="reader-lead">${escapeHtml(mdSnippet(methods, 280))}</p>
            <div class="hero-actions">${linkList(paper.links)}</div>
            ${figure}
          </header>

          <section class="reading-section" id="methods">
            <div class="reading-section-head">
              <p class="eyebrow">From Markdown Section 1</p>
              <h2>方法深度解析</h2>
            </div>
            <div class="markdown-body">${renderMarkdown(methods)}</div>
          </section>

          <section class="reading-section" id="experiments">
            <div class="reading-section-head">
              <p class="eyebrow">From Markdown Section 2</p>
              <h2>实验数据看板</h2>
            </div>
            ${experimentBoard(facts)}
            <div class="markdown-body">${renderMarkdown(experiments)}</div>
          </section>

          <section class="reading-section" id="visual-guides">
            <div class="reading-section-head">
              <p class="eyebrow">From Markdown Section 3</p>
              <h2>图文占位符</h2>
            </div>
            <div class="markdown-body">${renderMarkdown(visualGuides)}</div>
          </section>

          <section class="reading-section" id="limitations">
            <div class="reading-section-head">
              <p class="eyebrow">From Markdown Section 4</p>
              <h2>局限性与未来方向</h2>
            </div>
            <div class="markdown-body">${renderMarkdown(limitations)}</div>
          </section>

          <nav class="pager" aria-label="论文翻页">
            ${prev ? `<a href="${escapeHtml(localDetailHref(prev.detailPath))}">上一篇：${escapeHtml(prev.title)}</a>` : "<span></span>"}
            ${next ? `<a href="${escapeHtml(localDetailHref(next.detailPath))}">下一篇：${escapeHtml(next.title)}</a>` : "<span></span>"}
          </nav>
        </div>
      </article>
    `;
    hydrateMarkdown(this);
  }
}

customElements.define("literature-detail", LiteratureDetail);
