# UI-TARS-2: Advancing GUI Agent with Multi-Turn Reinforcement Learning (2025 - UI-TARS Team)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：UI-TARS-2 把 GUI agent 从“只能点屏幕的模型”升级成“能在统一沙盒里看 GUI、用终端、读写文件、调用工具、玩网页游戏，并用 PPO 做多轮 RL 的 computer-use agent”。

- **它到底解决什么问题**：第一代 GUI agent 主要卡在四件事上：数据难规模化、多轮 RL 不稳定、只会 GUI primitive 不会用文件/终端/tool、环境系统容易崩且吞吐低。UI-TARS-2 的回答不是一个单独算法，而是一整套训练和环境系统：Data Flywheel 持续生产 CT/SFT/RL 数据；All-in-One GUI Sandbox 支持 Windows/Ubuntu/Android/browser game；PPO 加 Decoupled-GAE、Length-Adaptive GAE、Value Pretraining、Clip Higher 稳定长轨迹训练；最后用参数插值把不同垂直 RL agent 合并成统一模型。

- **AI 怎么看（State）**：UI-TARS-2 仍然采用 native agent 视角，状态不是一张静态截图，而是 ReAct loop 中的历史上下文、当前 observation 和 memory：

$$
\tau=\{(t_0,a_0,o_0),(t_1,a_1,o_1),\ldots,(t_T,a_T,o_T)\}
$$

  其中 $t_t$ 是 reasoning / thought，$a_t$ 是动作，$o_t$ 是环境反馈。它还引入层级记忆：

$$
M_t=(W_t,E_t)
$$

  $W_t$ 是 working memory，保存最近若干步的高保真 thought/action/observation；$E_t$ 是 episodic memory，保存过去 episode 的压缩摘要、意图和结果。推理时策略预测：

$$
P(t_n,a_n \mid instruction, W_n, o_n, E_n)
$$

  所以 UI-TARS-2 的“状态”不是单张图片，而是“任务 + 当前界面/工具反馈 + 最近交互细节 + 长期摘要记忆”。

- **AI 怎么动（Action）**：动作空间分两层。第一层是 GUI actions，继承 UI-TARS：click、type、scroll、键盘/鼠标操作，游戏也复用这些 primitive。第二层是 GUI-SDK functions：终端命令、文件系统操作、代码相关工具、MCP/tool call、外部服务编排。也就是说，它不是纯鼠标键盘 agent，而是 GUI-centered computer-use agent：能先在浏览器下载文件，再用同一个容器里的 shell 处理文件，还能通过 proxy URL 预览 terminal 启动的前后端服务。

- **All-in-One GUI Sandbox 到底是什么**：

```text
Cloud VM sandbox:
  Windows / Ubuntu / Android
  PyAutoGUI 控制桌面鼠标键盘
  ADB 控制 Android
  VNC / RTC 实时可视化
  session ID 保持多轮任务状态
  shared file system 连接浏览器、终端、文件操作
  browser 里的 VS Code / Jupyter / terminal preview

Browser game sandbox:
  多个 Chrome/browser instance 跑在 container 中
  Chrome DevTools Protocol + Playwright compatible
  GPU hardware acceleration 降低截图开销
  fake clock / pause / resume / checkpoint 控制时间和复现
  crash recovery + garbage collector 保证长时间 rollout 稳定
```

  大白话说：它不是只开一个 Android emulator，也不是只跑一个 WebArena 浏览器，而是把桌面、手机、浏览器游戏、文件系统、终端和工具调用放进同一套可分配、可观察、可评估、可回收的 RL 环境。

- **Data Flywheel 是怎么转的**：UI-TARS-2 从 Seed1.6 checkpoint 出发，训练循环由 CT、SFT、RL 三段组成：

```text
初始数据：
  CT: GUI tutorial、instructional video、互联网演示、内部数据
  SFT: 合成任务 + 人工/交互标注的高质量轨迹

每轮迭代：
  1. 当前 RL model 产生新轨迹。
  2. validation function V(s) 判断样本好坏。
  3. V(s)=1 的高质量样本加入下一轮 SFT 数据。
  4. V(s)=0 的低质量但含知识样本进入 CT 数据。
  5. 模型按 CT -> SFT -> RL 顺序继续训练。
```

  这就是“飞轮”：模型越强，生成的好数据越多；好数据进入 SFT/RL 后又让模型更强。注意作者强调 SFT 和 RL 比 CT 更频繁，CT 负责广泛知识，SFT/RL 负责 agent-specific 能力。

- **RL 任务和 Reward 怎么设计**：UI-TARS-2 的 RLVR 覆盖三类代表任务。GUI-Browsing 是截图式信息搜索任务，有参考答案，用 LLM-as-Judge 判断答案是否匹配。GUI-General 是更开放的网页/GUI 操作，没有标准答案和函数 verifier，于是用 UI-TARS-2 自己增强成 generative ORM，输入完整文本历史和最后五张截图，输出成功分数。Gameplay 用 HTML5/WebGL 小游戏，JavaScript verifier 直接读取 score、level、lives 等运行时变量，给 scalar reward 和 termination flag。

  这三类 reward 分别对应三种可验证性：

| 任务类型 | verifier / reward |
|---|---|
| Gameplay | JS verifier 直接读游戏状态，函数式 reward |
| GUI-Browsing | reference answer + LLM-as-Judge |
| GUI-General | UI-TARS-2 generative ORM，看完整历史 + 最近 5 张截图 |

- **PPO 训练算法怎么改**：UI-TARS-2 最终选择 PPO，而不是 GRPO。论文说预实验里 PPO reward 更高、波动更低。基础 PPO objective 是：

$$
J_{PPO}(\theta)=
\mathbb{E}
\left[
\min
\left(
\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})}\hat{A}_t,
\text{clip}\left(
\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})},
1-\epsilon_{low},
1+\epsilon_{high}
\right)\hat{A}_t
\right)
\right]
$$

  为了让 PPO 能扛住 GUI 长轨迹，它加了四个稳定化技巧：

  - **Reward shaping**：主要看最终 outcome；部分场景加 format reward 和 length penalty，避免模型过早结束或无限拖延。
  - **Decoupled-GAE**：policy advantage 和 critic value target 使用不同的 $\lambda$，防止长 token 序列里 critic value 衰减/偏掉。
  - **Length-Adaptive GAE**：根据序列长度动态调 $\lambda_{policy}$，长序列用更接近 1 的 lambda，缓解不同长度轨迹 advantage 不一致。
  - **Value Pretraining**：先固定 SFT policy 采样轨迹，用 $\lambda=1.0$ 的 GAE，也就是近似 Monte Carlo return，把 value model 先训到稳定，再开始 PPO。作者观察到如果不做这一步，PPO 的 value estimate 甚至可能和真实 reward 负相关。
  - **Clip Higher**：把 PPO clip 分成 $\epsilon_{low}$ 和 $\epsilon_{high}$，提高上界让低概率但有价值的动作更容易被探索，同时保留下界避免 token diversity 过早坍缩。

- **垂直 agent 怎么合并**：联合训练所有环境很贵也不稳，因为 GUI-Browsing、GUI-General、Game、GUI-SDK 的动作/状态/轨迹长度差异巨大。UI-TARS-2 选择从同一个 SFT 初始化出发，分别训练多个 vertical RL agent，再做参数插值：

$$
\theta_{merge}=\sum_k \alpha_k\theta_k,\quad
\sum_k\alpha_k=1,\quad \alpha_k\ge 0
$$

  大白话：每个垂直 agent 学一个专长，最后不是再跑一次昂贵 multi-domain RL，而是在参数空间里把能力加权混合。

- 🎯**它解决了前人什么头疼的问题**：DigiRL/ARPO 这类方法证明了 GUI RL 可行，但通常只覆盖 Android、OSWorld 或某一类桌面任务；UI-TARS-2 要解决的是系统级 scale：环境要能跑、reward 要能验、数据要能循环、PPO 要能稳定、GUI-only 和 tool-use 要能合并。它的核心贡献不是“提出一个全新 RL 算法”，而是把多轮 GUI RL 所需的基础设施、数据飞轮、奖励体系和稳定 PPO 工程拼成一个可扩展系统。

## 2. 实验设置与结果 (Experiments)
- **基准测试 (Benchmarks)**：GUI 类包括 OSWorld、WindowsAgentArena、AndroidWorld、Online-Mind2Web、BrowseComp-zh/en。GUI-SDK 扩展后还测 Terminal Bench 和 SWE-Bench。游戏类包括内部 15 Games Collection 和 LMGame-Bench。OSWorld 有 369 个真实桌面任务；AndroidWorld 是 live Android emulator；WindowsAgentArena 评估 Windows GUI；Online-Mind2Web 评估 browser-use/web navigation；Terminal Bench/SWE-Bench 测终端和软件工程能力。

- **对比基线 (Baselines)**：GUI benchmark 对比 Claude-4-Sonnet、Claude-4-Opus、OpenAI o3、OpenAI CUA-o3、UI-TARS、UI-TARS-1.5。游戏 benchmark 对比 OpenAI CUA、Claude Computer Use、人类分数。论文还分析了 SFT vs RL、GUI-only vs GUI-SDK、PPO vs GRPO、Value Pretraining 有无等训练动态。

- **GUI / Computer-use 主结果**：

| 模型 | OSWorld | WAA | Terminal Bench | SWE-Bench | AndroidWorld | Online-Mind2Web | BrowseComp-zh | BrowseComp-en |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude-4-Sonnet | 43.9 | - | 39.2 | 72.7 | - | - | 22.5 | 14.7 |
| Claude-4-Opus | - | - | 43.2 | 72.5 | - | - | 37.4 | 18.8 |
| OpenAI o3 | 42.9 | - | 30.2 | 69.1 | 52.5 | 71.0 | - | 49.7 |
| OpenAI CUA-o3 | - | - | - | - | - | - | - | - |
| UI-TARS | 24.6 | - | - | - | 44.6 | - | - | - |
| UI-TARS-1.5 | 42.5 | 42.1 | - | - | 64.2 | 75.8 | - | - |
| UI-TARS-2 | 47.5 | 50.6 | 45.3† | 68.7† | 73.3 | 88.2 | 32.1 / 50.5† | 7.0 / 29.6† |

  `†` 表示使用扩展动作空间 GUI-SDK。最重要的读法是：UI-TARS-2 相比 UI-TARS-1.5 全面提升，OSWorld 从 42.5 到 47.5，WindowsAgentArena 到 50.6，AndroidWorld 从 64.2 到 73.3，Online-Mind2Web 从 75.8 到 88.2。GUI-SDK 对 BrowseComp 特别关键：中文从 32.1 提到 50.5，英文从 7.0 提到 29.6。

- **游戏结果**：在 15 Games Collection 上，人类归一化为 100，UI-TARS-2-SFT 为 44.27，UI-TARS-2-RL 为 59.77，OpenAI CUA 为 24.73，Claude Computer Use 为 21.61。RL 后模型接近人类 60% 平均水平，并在 Shapes 上超过人类，在 2048、Infinity-Loop、Tiles-master、Snake-solver、Merge-and-double 上接近人类较高比例。

| 方法 | Mean Normalized Score |
|---|---:|
| Human | 100.00 |
| UI-TARS-2-SFT | 44.27 |
| UI-TARS-2-RL | 59.77 |
| OpenAI CUA | 24.73 |
| Claude Computer Use | 21.61 |

- **消融实验 (Ablation Studies)**：第一，RL reward 曲线在 GUI-Browsing、GUI-General、Game 三类任务上都持续上升，说明 PPO 闭环对结构化 GUI 和动态游戏都有效。第二，ORM 存在误判但仍可用：作者构建 300 条人工标注 GUI trace 的 ORM eval set，UI-TARS-2 作为 binary ORM 的 F1 为 83.8；人工检查没有发现明显 reward hacking。第三，Value Pretraining 明显提高 PPO 稳定性；不预训练 value 时，value estimate 可能和实际 reward 负相关。第四，PPO vs GRPO：预实验中 PPO reward 更高、波动更低，因此最终选 PPO。第五，interaction scaling 显示给更多推理交互步数仍能提升 OSWorld/game 表现，说明 RL 后策略没有只学会短视操作。第六，hybrid GUI-only 与 GUI-SDK 训练显示，SDK 学到的高层工具能力可以迁移回 GUI-only，提高纯 GUI 操作稳定性；参数插值高效，hybrid RL 迁移更强但成本更高。

## 3. 图文占位符 (Visual Guides)
![UI-TARS-2 总体系统图](./images/uitars2_arch.png)

![All-in-One GUI Sandbox 架构图](./images/uitars2_sandbox.png)

![Data Flywheel 流程图](./images/uitars2_data_flywheel.png)

![Multi-turn PPO 训练框架](./images/uitars2_ppo_training.png)

![GUI-SDK 与共享文件系统示意图](./images/uitars2_gui_sdk.png)

![主 benchmark 结果表](./images/uitars2_results.png)

## 4. 局限性与未来方向 (Limitations)
- UI-TARS-2 是系统级技术报告，很多关键工程细节和数据构成无法完全复现。All-in-One Sandbox、数千 VM、内部任务生成、ORM 标注和 Data Flywheel 都需要大规模基础设施。
- Reward 仍然是瓶颈。GUI-General 依赖 generative ORM，F1 83.8 已经可用但仍有 false positive；长期 RL 里错误 reward 可能积累成 reward hacking。
- GUI-SDK 提升很大，但也改变了问题定义。使用终端、文件系统和 MCP/tool call 的结果不能和纯鼠标键盘 GUI-only agent 直接横向比较。
- 参数插值合并 vertical agents 高效但比较经验化，$\alpha_k$ 如何选择、能力冲突如何解决、不同 domain 间负迁移如何诊断，论文没有完全展开。
- PPO 稳定化技巧很多，说明多轮 GUI RL 仍然脆弱；Decoupled-GAE、Length-Adaptive GAE、Value Pretraining、Clip Higher 哪些是必要条件，仍需要更细消融。
- 未来方向包括：开源可复现的多平台 sandbox、更可靠的 step-level/process reward、GUI-only 与 tool-use 的公平评估协议，以及把 Data Flywheel 做成社区可运行的数据生成标准。
