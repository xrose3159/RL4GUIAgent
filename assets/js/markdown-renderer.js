import { siteUrl } from "./paper-data.js";

function escapeHtml(value = "") {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function imageToken(args) {
  if (typeof args === "object" && args !== null) return args;
  return { href: args, title: arguments[1], text: arguments[2] };
}

function linkToken(args) {
  if (typeof args === "object" && args !== null) return args;
  return { href: args, title: arguments[1], text: arguments[2] };
}

export function renderMarkdown(markdown = "") {
  if (!window.marked) {
    return `<pre><code>${escapeHtml(markdown)}</code></pre>`;
  }

  const renderer = new window.marked.Renderer();

  renderer.image = function (...args) {
    const token = imageToken(...args);
    const href = token.href || "";
    const alt = escapeHtml(token.text || "图片");
    if (href.startsWith("./images/")) {
      return `<div class="markdown-image-placeholder"><strong>${alt}</strong><br><code>${escapeHtml(href)}</code></div>`;
    }
    const src = /^https?:\/\//i.test(href) ? href : siteUrl(href.replace(/^\.\//, ""));
    const title = token.title ? ` title="${escapeHtml(token.title)}"` : "";
    return `<figure class="md-figure"><img src="${escapeHtml(src)}" alt="${alt}"${title} loading="lazy"><figcaption>${alt}</figcaption></figure>`;
  };

  renderer.link = function (...args) {
    const token = linkToken(...args);
    const href = token.href || "#";
    const label = token.text || href;
    const external = /^https?:\/\//i.test(href);
    const target = external ? ' target="_blank" rel="noreferrer"' : "";
    return `<a href="${escapeHtml(href)}"${target}>${label}</a>`;
  };

  const rawHtml = window.marked.parse(markdown, {
    gfm: true,
    breaks: false,
    renderer,
  });

  if (!window.DOMPurify) return rawHtml;
  return window.DOMPurify.sanitize(rawHtml, {
    ADD_TAGS: ["math"],
    ADD_ATTR: ["target", "rel", "loading"],
  });
}

export function hydrateMarkdown(root) {
  root.querySelectorAll("table").forEach((table) => table.classList.add("result-table"));
  root.querySelectorAll("pre code").forEach((code) => {
    code.parentElement.classList.add("code-block");
  });
  if (window.renderMathInElement) {
    window.renderMathInElement(root, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }
}
