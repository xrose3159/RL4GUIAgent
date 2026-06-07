const SECTION_KEYS = [
  ["methods", /核心方法|Methods/i],
  ["experiments", /实验设置|Experiments/i],
  ["visualGuides", /图文占位符|Visual Guides/i],
  ["limitations", /局限性|Limitations/i],
];

export function siteRoot() {
  return window.location.pathname.includes("/papers/") ? "../" : "./";
}

export function siteUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return new URL(`${siteRoot()}${path}`, document.baseURI).toString();
}

export async function fetchText(path) {
  const response = await fetch(siteUrl(path), { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.text();
}

export async function loadManifest() {
  const response = await fetch(siteUrl("data/papers.json"), { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`Failed to load data/papers.json: ${response.status}`);
  }
  return response.json();
}

export function parseMarkdownDocument(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const titleLine = lines.find((line) => line.startsWith("# ")) || "";
  const title = titleLine.replace(/^#\s+/, "").trim();
  const sections = {};
  let current = null;

  for (const line of lines) {
    if (line.startsWith("## ")) {
      const heading = line.replace(/^##\s+/, "").trim();
      const found = SECTION_KEYS.find(([, pattern]) => pattern.test(heading));
      current = found ? found[0] : heading;
      sections[current] = { heading, content: "" };
      continue;
    }
    if (current && !line.startsWith("# ")) {
      sections[current].content += `${line}\n`;
    }
  }

  return { title, sections, raw: markdown };
}

export async function loadPaperDocument(paper) {
  const markdown = await fetchText(paper.docPath);
  return { paper, document: parseMarkdownDocument(markdown) };
}

export function stripMarkdown(markdown = "") {
  return markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!\[[^\]]*]\([^)]+\)/g, " ")
    .replace(/\[([^\]]+)]\([^)]+\)/g, "$1")
    .replace(/[*_`>#|~-]/g, " ")
    .replace(/\$+/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function mdSnippet(markdown = "", max = 420) {
  const text = stripMarkdown(markdown);
  if (text.length <= max) return text;
  return `${text.slice(0, max).replace(/\s+\S*$/, "")}...`;
}

export function extractLabeledLine(markdown = "", labels = []) {
  const source = markdown.split("\n");
  for (const line of source) {
    const normalized = line.trim();
    if (!normalized.startsWith("-")) continue;
    for (const label of labels) {
      if (normalized.includes(label)) {
        return stripMarkdown(normalized.replace(/^-\s*/, "").replace(/^.*?：/, ""));
      }
    }
  }
  return "";
}

export function extractExperimentFacts(experimentMarkdown = "") {
  return [
    {
      label: "Benchmarks",
      text: extractLabeledLine(experimentMarkdown, ["基准测试", "Benchmarks"]),
    },
    {
      label: "Baselines",
      text: extractLabeledLine(experimentMarkdown, ["对比基线", "Baselines"]),
    },
    {
      label: "Metrics",
      text: extractLabeledLine(experimentMarkdown, ["核心量化指标"]),
    },
    {
      label: "Ablation",
      text: extractLabeledLine(experimentMarkdown, ["消融实验", "Ablation"]),
    },
  ].filter((item) => item.text);
}

export function deriveSignals(paper, document) {
  const text = `${paper.fullTitle} ${paper.tags?.join(" ") || ""} ${document.raw}`.toLowerCase();
  const environment = new Set(paper.environment || []);
  const hasSpecificEnvironment = [...environment].some((item) => item !== "Cross-platform GUI");
  if (!hasSpecificEnvironment) {
    environment.clear();
    if (/webarena|webshop|browser|网页|web /.test(text)) environment.add("Web");
    if (/android|aitw|mobile|手机|移动/.test(text)) environment.add("Android / Mobile");
    if (/osworld|desktop|windows|ubuntu|office|computer use|桌面/.test(text)) environment.add("Desktop / OS");
  }
  if (environment.size === 0) environment.add("Cross-platform GUI");

  const tags = paper.tags || [];
  let methodFamily = paper.methodFamily || "Framework / Benchmark";
  if (tags.includes("World Model")) {
    methodFamily = "World Model / Synthetic RL";
  } else if (tags.includes("Data")) {
    methodFamily = "Data Synthesis / Self-Evolution";
  } else if (tags.includes("Hybrid")) {
    methodFamily = "Hybrid Agent";
  } else if (tags.includes("Online RL")) {
    methodFamily = "Online RL";
  } else if (tags.includes("Offline RL")) {
    methodFamily = "Offline RFT / Offline RL";
  } else if (tags.includes("Grounding")) {
    methodFamily = "Grounding RL";
  } else if (tags.includes("Reward")) {
    methodFamily = "Reward Modeling";
  } else if (/world model|mcts|renderable code|synthetic/.test(text)) {
    methodFamily = "World Model / Synthetic RL";
  } else if (/hybrid action|api|mcp|hierarchical|planner|grounder|backtracking/.test(text)) {
    methodFamily = "Hybrid Agent";
  } else if (/grpo|ppo|awr|q-value|advantage|online rl/.test(text)) {
    methodFamily = "Offline RFT / Offline RL";
  }

  return { environment: [...environment], methodFamily };
}
