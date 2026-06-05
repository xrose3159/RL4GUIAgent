# GUI Agent RL 论文地图

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
