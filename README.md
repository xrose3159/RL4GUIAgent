# RL4GUIAgent

这是一个可直接部署到 GitHub Pages 的 GUI Agent RL 论文细读站点。

站点结构：

- `index.html`：总览、路线、搜索、筛选、环境/容器/框架总结。
- `papers/*.html`：41 篇论文独立细读页，包含 TL;DR、问题、任务形式、方法拆解、训练/奖励、实验、复现清单、局限。
- `build_site.py`：从上一级目录的 `gui_agent_rl_survey.md` 生成静态站点。

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
