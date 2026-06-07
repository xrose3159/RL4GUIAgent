# UI-S1: Semi-online Reinforcement Learning (2025 - Zhengxi Lu, Jiabo Ye, Fei Tang et al.)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：Semi-online RL 在离线 trajectory 中 rollout 当前模型输出，并用 Patch Module 修复模型动作偏离专家轨迹导致的 OOD 状态；奖励上同时用 discounted future return、step-level advantage 和 episode-level advantage；提出 SOP 作为 semi-online 性能代理指标。

- **这套系统是怎么运作的？（大白话工作流）**：离线 RL 有稳定和便宜的优点，但缺 trajectory-level 真实反馈；在线 RL 有反馈但代价高、奖励稀疏。需要在静态轨迹上模拟在线偏移和修正。 论文的做法可以理解成把 GUI agent 拆成“看状态、产动作、拿反馈、再更新”的闭环。与普通 SFT 最大的区别是，它不只学习专家下一步，而是把环境反馈、规则验证、reward model、搜索结果或世界模型预测变成可训练信号。

- **半在线：在离线轨迹里模拟当前策略偏离**：UI-S1 不真正访问环境，但也不只是静态 BC。它在离线专家轨迹中让当前模型 rollout，观察模型动作会怎样偏离专家路径。 问题是离线数据没有偏离后真实下一状态，所以 UI-S1 引入 Patch Module，把偏离动作对应的 OOD 状态修补回可训练状态。
- **Patch Module 的作用**：如果模型在某一步点错，真实 online 环境会进入新页面；离线数据里没有这个页面。Patch Module 尝试根据已有轨迹和任务结构构造一个可继续训练的状态，使半在线 rollout 不会立刻断掉。 这样模型能学习“如果自己偏离了专家，该如何回到正确路径附近”，比纯离线行为克隆更接近部署。
- **三层奖励**：UI-S1 同时用 discounted future return、step-level advantage 和 episode-level advantage。step 信号判断当前动作是否推进，episode 信号判断整条轨迹是否好。 SOP 指标用于估计 semi-online 训练质量，帮助在没有真实 online 环境的情况下选择模型。

- **AI 怎么看（状态输入 State）**：离线轨迹中的截图状态 + 当前模型模拟动作 + Patch Module 修复后的状态。

- **AI 怎么想（决策中心 Controller）**：混合训练流程：通常先 SFT/数据冷启动，再用规则 reward、VLM judge、process reward 或 replay/filtering 做 RFT/RL/自演化。 这里要抓住一个关键点：GUI agent 的 reward 往往延迟且稀疏，所以论文一定要用某种办法把“最后成功/失败”拆成更可学的信号，例如组内相对优势、step-level progress、成功轨迹 replay、world-model lookahead、任务难度 curriculum 或 chosen/rejected 偏好对。

- **AI 怎么动（动作空间 Action）**：GUI action prediction/trajectory action。

- **Reward / 训练信号怎么给**：discounted future return、step-level advantage、episode-level advantage，并用 SOP 估计 semi-online 训练质量。

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
- **基准测试 (Benchmarks)**：AndroidWorld、AITW 等动态 benchmark。

- **对比基线 (Baselines)**：通常包括同尺寸 SFT/BC 模型、prompting agent、闭源 VLM/LLM、已有 GUI grounding/agent 模型，以及去掉核心模块的 ablation 版本。具体到本文，比较重点是证明 `UI-S1` 这一路线相对纯 SFT、纯 prompting 或朴素 RL/RFT 的收益。

- **核心量化指标**：7B 模型在四个动态 benchmark 达 SOTA；相比 base model，AndroidWorld +12.0%，AITW +23.8%。这是“用离线吞吐模拟在线训练”的代表。

| 项目 | 论文口径 | 读法 |
|---|---|---|
| 主指标 | 任务成功率、grounding accuracy、平均 reward、轨迹成本或系统吞吐 | 不同 benchmark 口径不能直接横比，必须看环境和动作空间。 |
| 训练信号 | discounted future return、step-level advantage、episode-level advantage，并用 SOP 估计 semi-online 训练质量。 | 判断它是终局 reward、过程 reward、偏好信号还是世界模型反馈。 |
| 环境 | AndroidWorld、AITW 等动态 benchmark。 | 看是否真实交互、可重置、是否依赖 DOM/HTML/ADB/VM/browser sandbox。 |
| 模型/基线 | SFT、BC、prompting、闭源模型、同尺寸开源模型和消融项 | 重点看是否公平使用同一 backbone、同一任务和同一 step budget。 |

- **消融实验 (Ablation Studies)**：应重点关注四类消融：去掉 reward/verifier 后是否退化；去掉 replay/curriculum/filtering 后是否训练不稳定；去掉 reasoning/tool/world-model 后是否长任务失败；改变数据规模、候选数、环境并行数或 step budget 后性能如何变化。若论文报告系统吞吐，还要同时看 rollout latency、GPU 利用率、VM/浏览器并行数和每条轨迹成本。

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/ui_s1_arch.png)

![训练流程图](./images/ui_s1_training.png)

![Reward 或数据流示意图](./images/ui_s1_reward.png)

![实验结果表](./images/ui_s1_results.png)

## 4. 局限性与未来方向 (Limitations)
- 这类方法的结论通常强依赖 benchmark 和环境接口。Web/Android/Desktop/OSWorld/ScreenSpot 的状态、动作和 reward 都不同，不能把一个表里的成功率直接推广到所有 GUI 任务。
- 如果 reward 来自 VLM/ORM judge，误判会污染训练；如果 reward 来自规则 verifier，则任务覆盖常受限于可写规则的场景。
- 在线 RL 和世界模型路线都需要大量系统工程：可重置环境、并行 rollout、稳定截图/DOM/ADB/VM、replay buffer、失败恢复和日志审计。
- 对 grounding 论文而言，单步坐标准确率不等于完整任务成功率；对 computer-use 论文而言，允许 GUI-SDK/API/tool call 后，和纯 GUI-only agent 的比较要分开看。
- 未来方向是更可靠的过程奖励、更统一的跨平台 action schema、更低成本的环境/世界模型、更强的错误恢复，以及把学习到的技能长期保存为可复用记忆或工具。
