# GELab-Zero / GUI Exploration Lab (2025 - Haolong Yan, Yeqing Shen, Xin Huang et al.)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：构建 GUI Exploration Lab，一个 GUI 导航模拟环境引擎，可灵活定义 screen、icon、navigation graph，并提供完整环境信息。训练上比较 SFT、single-turn RL 和 multi-turn RL 的作用。

- **这套系统是怎么运作的？（大白话工作流）**：真实 PC 软件和移动 App 多为闭源，难以获得完整环境信息，导致 GUI 导航训练和评测不系统。 论文的做法可以理解成把 GUI agent 拆成“看状态、产动作、拿反馈、再更新”的闭环。与普通 SFT 最大的区别是，它不只学习专家下一步，而是把环境反馈、规则验证、reward model、搜索结果或世界模型预测变成可训练信号。

- **先做可控 GUI 环境，再谈 RL**：GELab-Zero / GUI Exploration Lab 的重点是环境引擎。它允许定义 screen、icon、navigation graph 和完整环境状态，使 GUI 导航任务可重置、可验证、可探索。 这解决了真实软件闭源、状态不可控、任务难复现的问题。
- **比较 SFT、single-turn RL、multi-turn RL**：在这个环境里，SFT 负责记住基础导航知识，single-turn RL 强化未见场景下的单步泛化，multi-turn RL 让模型通过交互试错学习探索策略。 这种实验设计把不同训练方式的作用拆开，而不是只报告一个最终成功率。
- **为什么叫 Zero**：它的目标是降低人工轨迹依赖：任务和反馈尽量由环境状态自动给出，模型通过探索积累成功/失败经验。 所以 GELab 更像 GUI RL 的平台论文，为后续 mobile/GUI self-evolution 提供可控场地。

- **AI 怎么看（状态输入 State）**：模拟 GUI navigation graph 中的 screen/icon/environment metadata。

- **AI 怎么想（决策中心 Controller）**：混合训练流程：通常先 SFT/数据冷启动，再用规则 reward、VLM judge、process reward 或 replay/filtering 做 RFT/RL/自演化。 这里要抓住一个关键点：GUI agent 的 reward 往往延迟且稀疏，所以论文一定要用某种办法把“最后成功/失败”拆成更可学的信号，例如组内相对优势、step-level progress、成功轨迹 replay、world-model lookahead、任务难度 curriculum 或 chosen/rejected 偏好对。

- **AI 怎么动（动作空间 Action）**：点击/导航到下一个 screen；可定义探索策略。

- **Reward / 训练信号怎么给**：环境知道完整 navigation graph，可判定是否达到目标 screen/状态。

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
- **基准测试 (Benchmarks)**：GELab-Zero 静态与交互 benchmark，迁移到真实 GUI 场景。

- **对比基线 (Baselines)**：通常包括同尺寸 SFT/BC 模型、prompting agent、闭源 VLM/LLM、已有 GUI grounding/agent 模型，以及去掉核心模块的 ablation 版本。具体到本文，比较重点是证明 `GELab` 这一路线相对纯 SFT、纯 prompting 或朴素 RL/RFT 的收益。

- **核心量化指标**：作者发现 SFT 记忆基础知识，single-turn RL 增强未见场景泛化，multi-turn RL 通过交互试错学习探索策略。静态和交互 benchmark 均验证 RL 对 GUI navigation 有收益，并可泛化到真实场景。

| 项目 | 论文口径 | 读法 |
|---|---|---|
| 主指标 | 任务成功率、grounding accuracy、平均 reward、轨迹成本或系统吞吐 | 不同 benchmark 口径不能直接横比，必须看环境和动作空间。 |
| 训练信号 | 环境知道完整 navigation graph，可判定是否达到目标 screen/状态。 | 判断它是终局 reward、过程 reward、偏好信号还是世界模型反馈。 |
| 环境 | GELab-Zero 静态与交互 benchmark，迁移到真实 GUI 场景。 | 看是否真实交互、可重置、是否依赖 DOM/HTML/ADB/VM/browser sandbox。 |
| 模型/基线 | SFT、BC、prompting、闭源模型、同尺寸开源模型和消融项 | 重点看是否公平使用同一 backbone、同一任务和同一 step budget。 |

- **消融实验 (Ablation Studies)**：应重点关注四类消融：去掉 reward/verifier 后是否退化；去掉 replay/curriculum/filtering 后是否训练不稳定；去掉 reasoning/tool/world-model 后是否长任务失败；改变数据规模、候选数、环境并行数或 step budget 后性能如何变化。若论文报告系统吞吐，还要同时看 rollout latency、GPU 利用率、VM/浏览器并行数和每条轨迹成本。

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/gelab_arch.png)

![训练流程图](./images/gelab_training.png)

![Reward 或数据流示意图](./images/gelab_reward.png)

![实验结果表](./images/gelab_results.png)

## 4. 局限性与未来方向 (Limitations)
- 这类方法的结论通常强依赖 benchmark 和环境接口。Web/Android/Desktop/OSWorld/ScreenSpot 的状态、动作和 reward 都不同，不能把一个表里的成功率直接推广到所有 GUI 任务。
- 如果 reward 来自 VLM/ORM judge，误判会污染训练；如果 reward 来自规则 verifier，则任务覆盖常受限于可写规则的场景。
- 在线 RL 和世界模型路线都需要大量系统工程：可重置环境、并行 rollout、稳定截图/DOM/ADB/VM、replay buffer、失败恢复和日志审计。
- 对 grounding 论文而言，单步坐标准确率不等于完整任务成功率；对 computer-use 论文而言，允许 GUI-SDK/API/tool call 后，和纯 GUI-only agent 的比较要分开看。
- 未来方向是更可靠的过程奖励、更统一的跨平台 action schema、更低成本的环境/世界模型、更强的错误恢复，以及把学习到的技能长期保存为可复用记忆或工具。
