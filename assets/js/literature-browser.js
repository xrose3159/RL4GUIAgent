import {
  deriveSignals,
  extractExperimentFacts,
  loadManifest,
  loadPaperDocument,
  mdSnippet,
  stripMarkdown,
} from "./paper-data.js";
import { hydrateMarkdown, renderMarkdown } from "./markdown-renderer.js";

function escapeHtml(value = "") {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function countBy(items, key) {
  const counts = new Map();
  for (const item of items) {
    const values = Array.isArray(item[key]) ? item[key] : [item[key]];
    for (const value of values.filter(Boolean)) {
      counts.set(value, (counts.get(value) || 0) + 1);
    }
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

function firstMethodBlock(markdown) {
  const lines = markdown
    .split("\n")
    .filter((line) => line.trim() && !line.trim().startsWith("###"));
  return mdSnippet(lines.slice(0, 12).join("\n"), 520);
}

function boardHtml(facts) {
  if (!facts.length) return `<p class="paper-empty">MD 第 2 部分没有抽取到显式实验标签，进入详情页可读完整实验段落。</p>`;
  return facts
    .slice(0, 3)
    .map(
      (fact) => `<div>
        <span>${escapeHtml(fact.label)}</span>
        <p>${escapeHtml(fact.text)}</p>
      </div>`,
    )
    .join("");
}

class LiteratureBrowser extends HTMLElement {
  async connectedCallback() {
    this.state = { query: "", filter: "all", filterType: "all" };
    this.innerHTML = `<section class="reader-state"><p class="eyebrow">Loading Markdown</p><h2>正在读取 41 篇 MD 文献...</h2></section>`;
    try {
      const manifest = await loadManifest();
      const loaded = await Promise.all((manifest.papers || []).map((paper) => loadPaperDocument(paper)));
      this.items = loaded.map(({ paper, document }) => {
        const signals = deriveSignals(paper, document);
        const methods = document.sections.methods?.content || "";
        const experiments = document.sections.experiments?.content || "";
        return {
          ...paper,
          document,
          signals,
          methods,
          experiments,
          methodPreview: firstMethodBlock(methods),
          experimentFacts: extractExperimentFacts(experiments),
          searchText: stripMarkdown(`${paper.fullTitle} ${paper.tags.join(" ")} ${signals.methodFamily} ${signals.environment.join(" ")} ${methods} ${experiments}`).toLowerCase(),
        };
      });
      this.render();
    } catch (error) {
      this.innerHTML = `<section class="reader-state"><p class="eyebrow">Load failed</p><h2>文献数据库加载失败</h2><p>${escapeHtml(error.message)}</p></section>`;
    }
  }

  filteredItems() {
    return this.items.filter((item) => {
      const queryOk = !this.state.query || item.searchText.includes(this.state.query);
      if (!queryOk) return false;
      if (this.state.filter === "all") return true;
      if (this.state.filterType === "method") return item.signals.methodFamily === this.state.filter;
      if (this.state.filterType === "environment") return item.signals.environment.includes(this.state.filter);
      return item.tags.includes(this.state.filter);
    });
  }

  setFilter(filterType, filter) {
    this.state.filterType = filterType;
    this.state.filter = filter;
    this.renderCards();
  }

  render() {
    const methodCounts = countBy(this.items.map((item) => item.signals), "methodFamily");
    const envCounts = countBy(this.items.map((item) => item.signals), "environment");
    const tagCounts = countBy(this.items, "tags");
    this.innerHTML = `
      <div class="library-shell">
        <aside class="library-rail">
          <div class="library-stamp">
            <span>Source-bound</span>
            <strong>docs/*.md</strong>
          </div>
          <label class="search-box">
            <span>搜索</span>
            <input id="library-search" type="search" placeholder="论文、算法、环境、benchmark">
          </label>
          ${this.renderFilterGroup("方法路线", "method", methodCounts)}
          ${this.renderFilterGroup("GUI 环境", "environment", envCounts)}
          ${this.renderFilterGroup("标签", "tag", tagCounts)}
        </aside>
        <section class="library-main">
          <div class="library-heading">
            <div>
              <p class="eyebrow">Markdown Corpus</p>
              <h2>41 篇文献精读库</h2>
              <p>首页卡片的“方法深度解析”和“实验数据看板”来自对应 MD 的第 1、2 部分；详情页会完整渲染这两部分。</p>
            </div>
            <div class="library-count"><strong id="library-count">41</strong><span>papers</span></div>
          </div>
          <div id="library-cards" class="deep-paper-list"></div>
        </section>
      </div>
    `;
    this.querySelector("#library-search").addEventListener("input", (event) => {
      this.state.query = event.target.value.trim().toLowerCase();
      this.renderCards();
    });
    this.querySelectorAll("[data-filter]").forEach((button) => {
      button.addEventListener("click", () => {
        this.setFilter(button.dataset.filterType, button.dataset.filter);
      });
    });
    this.renderCards();
  }

  renderFilterGroup(title, type, entries) {
    return `<section class="filter-group">
      <h3>${escapeHtml(title)}</h3>
      <button class="filter-chip" data-filter-type="all" data-filter="all">全部 <span>${this.items.length}</span></button>
      ${entries
        .map(([label, count]) => `<button class="filter-chip" data-filter-type="${escapeHtml(type)}" data-filter="${escapeHtml(label)}">${escapeHtml(label)} <span>${count}</span></button>`)
        .join("")}
    </section>`;
  }

  renderCards() {
    const cards = this.filteredItems();
    const count = this.querySelector("#library-count");
    const target = this.querySelector("#library-cards");
    if (count) count.textContent = String(cards.length);
    target.innerHTML = cards.map((item) => this.renderCard(item)).join("");
    hydrateMarkdown(target);
  }

  renderCard(item) {
    return `<article class="paper-deep-card">
      <header>
        <a class="paper-no" href="${escapeHtml(item.detailPath)}">${escapeHtml(item.id)}</a>
        <div>
          <h3><a href="${escapeHtml(item.detailPath)}">${escapeHtml(item.title)}</a></h3>
          <div class="tag-row">
            <span class="tag">${escapeHtml(item.signals.methodFamily)}</span>
            ${item.signals.environment.map((env) => `<span class="tag">${escapeHtml(env)}</span>`).join("")}
          </div>
        </div>
      </header>
      <div class="paper-source-line">直接读取 <code>${escapeHtml(item.docPath)}</code></div>
      <div class="paper-insight-grid">
        <section>
          <h4>方法深度解析 <span>MD §1</span></h4>
          <p>${escapeHtml(item.methodPreview)}</p>
        </section>
        <section>
          <h4>实验数据看板 <span>MD §2</span></h4>
          <div class="mini-board">${boardHtml(item.experimentFacts)}</div>
        </section>
      </div>
      <details>
        <summary>展开 MD 第 1 部分原文片段</summary>
        <div class="markdown-body">${renderMarkdown(item.methods.split("\n").slice(0, 36).join("\n"))}</div>
      </details>
      <div class="card-readmore"><a href="${escapeHtml(item.detailPath)}">进入完整 MD 渲染详情页</a></div>
    </article>`;
  }
}

customElements.define("literature-browser", LiteratureBrowser);
