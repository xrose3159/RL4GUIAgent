# RL4GUIAgent

这是一个可直接部署到 GitHub Pages 的 GUI Agent RL 论文细读站点。

站点结构：

- `docs/*.md`：41 篇论文的权威 Markdown 精读数据源。
- `data/papers.json`：论文元数据、标签、环境、方法路线、详情页路径和 MD 路径。
- `index.html`：总览、路线、环境/容器/框架总结；论文库由 `<literature-browser>` 运行时读取 `docs/*.md` 渲染。
- `papers/*.html`：41 篇论文独立详情页壳；正文由 `<literature-detail>` 运行时读取对应 MD 文档。
- `assets/js/`：Markdown 数据读取、Markdown/LaTeX 渲染、文献浏览器和文献详情组件。
- `build_site.py`：旧版静态站点生成脚本，保留作参考；当前正文不再依赖它的粗粒度摘要。

详情页中：

- “方法深度解析”直接渲染 MD 第 1 部分。
- “实验数据看板”从 MD 第 2 部分抽取 benchmark、baseline、metric、ablation，并继续完整渲染第 2 部分原文。
- Markdown 渲染使用 `marked + DOMPurify + KaTeX`，支持 GFM 表格、加粗、列表、代码块和 LaTeX 公式。

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

仓库地址：

```bash
git@github.com:xrose3159/RL4GUIAgent.git
```

推送后在 GitHub 仓库设置中开启 Pages：

- Source: Deploy from a branch
- Branch: `main`
- Folder: `/root`

发布成功后，站点地址通常为：

```text
https://xrose3159.github.io/RL4GUIAgent/
```
