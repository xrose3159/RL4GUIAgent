# Hi-Agent: Hierarchical Vision-Language Agents for Mobile Device Control (2025 - Zhe Wu, Hongjin Lu, Junliang Xing et al.)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：Hi-Agent 把移动 GUI agent 拆成高层 reasoning model 与低层 action model，并联合优化。高层负责分解任务、生成中间目标和上下文决策；低层根据当前屏幕和子目标执行具体 GUI 动作。训练上属于 hybrid/层级路线：用示范冷启动，再用 RL 强化高低层协同，缓解长时序 credit assignment。

- **这套系统是怎么运作的？（大白话工作流）**：移动端任务天然有层级结构：高层需要规划“先打开哪个 app、进入哪个页面、完成哪个子目标”，低层需要精确执行点击、输入和滑动。单一模型直接从指令到动作，容易在长任务中丢失全局目标，也难以同时兼顾语义推理和坐标/控件级执行。 论文的做法可以理解成把 GUI agent 拆成“看状态、产动作、拿反馈、再更新”的闭环。与普通 SFT 最大的区别是，它不只学习专家下一步，而是把环境反馈、规则验证、reward model、搜索结果或世界模型预测变成可训练信号。

- **高层规划和低层执行分开**：Hi-Agent 把移动 GUI 控制拆成两层。高层 reasoning model 理解任务、拆子目标、决定下一阶段；低层 action model 根据当前截图和子目标输出 tap/type/swipe 等具体动作。 例如高层说“进入账号安全设置”，低层才负责在当前屏幕上点哪个按钮、是否滚动、输入什么文本。
- **为什么层级有用**：长手机任务中，单一模型既要记全局目标又要精确点控件，容易丢失上下文。层级结构让高层处理语义和规划，低层处理视觉定位和动作执行。 这也降低 credit assignment 难度：高层错误和低层错误可以分开分析。
- **训练和协同**：训练通常先用示范轨迹让两层对齐，再用 RL 或反馈强化协同。高层目标如果不可执行，低层会失败；低层误点也会破坏高层计划。 所以 Hi-Agent 的重点不是单个 reward 公式，而是移动 GUI 任务的层级控制范式。

- **AI 怎么看（状态输入 State）**：高层看到任务、历史和当前屏幕；低层看到当前屏幕和高层子目标。

- **AI 怎么想（决策中心 Controller）**：混合训练流程：通常先 SFT/数据冷启动，再用规则 reward、VLM judge、process reward 或 replay/filtering 做 RFT/RL/自演化。 这里要抓住一个关键点：GUI agent 的 reward 往往延迟且稀疏，所以论文一定要用某种办法把“最后成功/失败”拆成更可学的信号，例如组内相对优势、step-level progress、成功轨迹 replay、world-model lookahead、任务难度 curriculum 或 chosen/rejected 偏好对。

- **AI 怎么动（动作空间 Action）**：高层输出子目标/计划，低层输出 tap/type/swipe/back 等移动端动作。

- **Reward / 训练信号怎么给**：任务成功与高低层协同 reward；用 RL 缓解长任务 credit assignment。

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
- **基准测试 (Benchmarks)**：AITW、AndroidWorld、AndroidControl-Low/High、AMEX、AndroidLab。

- **对比基线 (Baselines)**：通常包括同尺寸 SFT/BC 模型、prompting agent、闭源 VLM/LLM、已有 GUI grounding/agent 模型，以及去掉核心模块的 ablation 版本。具体到本文，比较重点是证明 `Hi-Agent` 这一路线相对纯 SFT、纯 prompting 或朴素 RL/RFT 的收益。

- **核心量化指标**：论文摘要报告 Hi-Agent 在 AITW 达 87.9%，AndroidWorld 达 21.0%，并在 AndroidControl-Low/High、AMEX、AndroidLab 等基准上展示跨任务泛化。综述把它作为“System-1 执行 + System-2 规划”式认知分层的代表。

| 项目 | 论文口径 | 读法 |
|---|---|---|
| 主指标 | 任务成功率、grounding accuracy、平均 reward、轨迹成本或系统吞吐 | 不同 benchmark 口径不能直接横比，必须看环境和动作空间。 |
| 训练信号 | 任务成功与高低层协同 reward；用 RL 缓解长任务 credit assignment。 | 判断它是终局 reward、过程 reward、偏好信号还是世界模型反馈。 |
| 环境 | AITW、AndroidWorld、AndroidControl-Low/High、AMEX、AndroidLab。 | 看是否真实交互、可重置、是否依赖 DOM/HTML/ADB/VM/browser sandbox。 |
| 模型/基线 | SFT、BC、prompting、闭源模型、同尺寸开源模型和消融项 | 重点看是否公平使用同一 backbone、同一任务和同一 step budget。 |

- **消融实验 (Ablation Studies)**：应重点关注四类消融：去掉 reward/verifier 后是否退化；去掉 replay/curriculum/filtering 后是否训练不稳定；去掉 reasoning/tool/world-model 后是否长任务失败；改变数据规模、候选数、环境并行数或 step budget 后性能如何变化。若论文报告系统吞吐，还要同时看 rollout latency、GPU 利用率、VM/浏览器并行数和每条轨迹成本。

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/hi_agent_arch.png)

![训练流程图](./images/hi_agent_training.png)

![Reward 或数据流示意图](./images/hi_agent_reward.png)

![实验结果表](./images/hi_agent_results.png)

## 4. 局限性与未来方向 (Limitations)
- 这类方法的结论通常强依赖 benchmark 和环境接口。Web/Android/Desktop/OSWorld/ScreenSpot 的状态、动作和 reward 都不同，不能把一个表里的成功率直接推广到所有 GUI 任务。
- 如果 reward 来自 VLM/ORM judge，误判会污染训练；如果 reward 来自规则 verifier，则任务覆盖常受限于可写规则的场景。
- 在线 RL 和世界模型路线都需要大量系统工程：可重置环境、并行 rollout、稳定截图/DOM/ADB/VM、replay buffer、失败恢复和日志审计。
- 对 grounding 论文而言，单步坐标准确率不等于完整任务成功率；对 computer-use 论文而言，允许 GUI-SDK/API/tool call 后，和纯 GUI-only agent 的比较要分开看。
- 未来方向是更可靠的过程奖励、更统一的跨平台 action schema、更低成本的环境/世界模型、更强的错误恢复，以及把学习到的技能长期保存为可复用记忆或工具。
