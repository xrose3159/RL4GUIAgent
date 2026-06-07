# Code2World: GUI World Model via Renderable Code Generation (2026 - Yuhao Zheng, Li'an Zhong, Yi Wang et al.)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：把 next visual state 预测转成 renderable code generation。构造 AndroidCode，把 GUI 轨迹翻译成高保真 HTML，并用视觉反馈 revision；训练先 SFT 冷启动格式与布局，再用 Render-Aware RL，以渲染结果的视觉语义一致性和动作一致性为 reward。

- **这套系统是怎么运作的？（大白话工作流）**：GUI 世界模型需要既有高视觉保真，又有结构可控。纯文本或纯像素预测往往不能同时满足。 论文的做法可以理解成把 GUI agent 拆成“看状态、产动作、拿反馈、再更新”的闭环。与普通 SFT 最大的区别是，它不只学习专家下一步，而是把环境反馈、规则验证、reward model、搜索结果或世界模型预测变成可训练信号。

- **把下一屏预测变成可渲染代码生成**：Code2World 不直接生成像素图，而是生成可渲染代码，例如 HTML/CSS/布局结构，再渲染成下一 UI。这样状态更可解释，也更容易检查和编辑。 它构造 AndroidCode，把 GUI 轨迹转换成高保真 HTML，并通过视觉反馈 revision 改善渲染质量。
- **Render-Aware RL**：训练先用 SFT 学会格式和布局，再用 Render-Aware RL。reward 来自渲染后的视觉语义一致性、动作一致性和下一状态是否符合真实轨迹。 模型因此学的不只是页面长什么样，还要学动作会怎样改变界面。
- **为什么结构化世界模型有优势**：代码状态可以被 diff、验证和复用，比黑箱图像生成更适合作为 agent 训练环境。 局限是 GUI 中很多动态效果、原生控件和复杂交互不容易用 HTML 完整复刻。

- **AI 怎么看（状态输入 State）**：当前 GUI 截图/动作和生成的 renderable code 状态。

- **AI 怎么想（决策中心 Controller）**：世界模型/模型式 RL：先学习环境转移或可渲染状态，再在模型里做 rollout、搜索或轨迹合成，降低真实 GUI I/O 成本。 这里要抓住一个关键点：GUI agent 的 reward 往往延迟且稀疏，所以论文一定要用某种办法把“最后成功/失败”拆成更可学的信号，例如组内相对优势、step-level progress、成功轨迹 replay、world-model lookahead、任务难度 curriculum 或 chosen/rejected 偏好对。

- **AI 怎么动（动作空间 Action）**：生成下一 UI 的 HTML/CSS/布局代码，或用世界模型辅助导航。

- **Reward / 训练信号怎么给**：Render-Aware RL：渲染结果的视觉语义一致性 + 动作一致性。

- **训练流程伪代码**：

```text
1. 准备任务集合、GUI/Web/OS 环境或离线轨迹数据。
2. 用基础 VLM/LLM agent 读取任务和状态，生成动作或多条候选轨迹。
3. 环境、规则 verifier、reward model、搜索器或世界模型给出成功/失败、进度、坐标命中或偏好信号。
4. 过滤无效样本，保留成功轨迹、困难任务、正 advantage 步骤或高质量 synthetic trajectories。
5. 用对应优化方法更新策略：SFT/RFT、GRPO、PPO、AWR/DPO、world-model rollout 或混合训练。
6. 如果是在线/自演化方法，把新轨迹写回 replay/data pool，继续下一轮任务采样和训练。
```

- 🎯**它解决了前人什么头疼的问题？**：以前的 GUI agent 多数只靠静态示范或 prompt，容易出现三个问题：界面变化后状态分布漂移、长任务终局奖励太稀疏、单步动作正确但整条任务失败。本文的价值在于把这些问题转成一个更可训练的机制：要么让奖励更密，要么让任务更可学，要么让搜索/世界模型产生更多好轨迹，要么让环境系统足够稳定地支撑大规模 rollout。

## 2. 实验设置与结果 (Experiments)
- **基准测试 (Benchmarks)**：AndroidCode next UI prediction，下游导航辅助评估。

- **对比基线 (Baselines)**：通常包括同尺寸 SFT/BC 模型、prompting agent、闭源 VLM/LLM、已有 GUI grounding/agent 模型，以及去掉核心模块的 ablation 版本。具体到本文，比较重点是证明 `Code2World` 这一路线相对纯 SFT、纯 prompting 或朴素 RL/RFT 的收益。

- **核心量化指标**：AndroidCode 超 80K 高质量 screen-action pairs；Code2World-8B 在 next UI prediction 上表现顶级，接近 GPT-5 和 Gemini-3-Pro-Image；作为下游导航辅助，使 Gemini-2.5-Flash 在 AndroidWorld navigation +9.5%。

| 项目 | 论文口径 | 读法 |
|---|---|---|
| 主指标 | 任务成功率、grounding accuracy、平均 reward、轨迹成本或系统吞吐 | 不同 benchmark 口径不能直接横比，必须看环境和动作空间。 |
| 训练信号 | Render-Aware RL：渲染结果的视觉语义一致性 + 动作一致性。 | 判断它是终局 reward、过程 reward、偏好信号还是世界模型反馈。 |
| 环境 | AndroidCode next UI prediction，下游导航辅助评估。 | 看是否真实交互、可重置、是否依赖 DOM/HTML/ADB/VM/browser sandbox。 |
| 模型/基线 | SFT、BC、prompting、闭源模型、同尺寸开源模型和消融项 | 重点看是否公平使用同一 backbone、同一任务和同一 step budget。 |

- **消融实验 (Ablation Studies)**：应重点关注四类消融：去掉 reward/verifier 后是否退化；去掉 replay/curriculum/filtering 后是否训练不稳定；去掉 reasoning/tool/world-model 后是否长任务失败；改变数据规模、候选数、环境并行数或 step budget 后性能如何变化。若论文报告系统吞吐，还要同时看 rollout latency、GPU 利用率、VM/浏览器并行数和每条轨迹成本。

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/code2world_arch.png)

![训练流程图](./images/code2world_training.png)

![Reward 或数据流示意图](./images/code2world_reward.png)

![实验结果表](./images/code2world_results.png)

## 4. 局限性与未来方向 (Limitations)
- 这类方法的结论通常强依赖 benchmark 和环境接口。Web/Android/Desktop/OSWorld/ScreenSpot 的状态、动作和 reward 都不同，不能把一个表里的成功率直接推广到所有 GUI 任务。
- 如果 reward 来自 VLM/ORM judge，误判会污染训练；如果 reward 来自规则 verifier，则任务覆盖常受限于可写规则的场景。
- 在线 RL 和世界模型路线都需要大量系统工程：可重置环境、并行 rollout、稳定截图/DOM/ADB/VM、replay buffer、失败恢复和日志审计。
- 对 grounding 论文而言，单步坐标准确率不等于完整任务成功率；对 computer-use 论文而言，允许 GUI-SDK/API/tool call 后，和纯 GUI-only agent 的比较要分开看。
- 未来方向是更可靠的过程奖励、更统一的跨平台 action schema、更低成本的环境/世界模型、更强的错误恢复，以及把学习到的技能长期保存为可复用记忆或工具。
