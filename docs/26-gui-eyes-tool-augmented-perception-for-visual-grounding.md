# GUI-Eyes: Tool-Augmented Perception for Visual Grounding (2026 - Chen Chen, Jiawei Shao, Dakuan Lu et al.)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：用 GRPO 学习主动视觉工具调用。两阶段推理：coarse exploration 判断要不要 crop/zoom，fine-grained grounding 输出坐标；奖励用连续空间函数，结合位置接近度和区域重叠。

- **这套系统是怎么运作的？（大白话工作流）**：多数 GUI grounding 方法只看一次静态全屏，缺少主动决定“是否放大、裁剪、重看”的能力。 论文的做法可以理解成把 GUI agent 拆成“看状态、产动作、拿反馈、再更新”的闭环。与普通 SFT 最大的区别是，它不只学习专家下一步，而是把环境反馈、规则验证、reward model、搜索结果或世界模型预测变成可训练信号。

- **主动感知：先决定要不要重看**：GUI-Eyes 认为很多 grounding 错误不是策略不会，而是全屏截图看不清小图标、小文字或密集控件。它让模型学会调用视觉工具，例如 crop、zoom、局部重看。 推理分两阶段：coarse exploration 找可能区域和是否需要工具，fine-grained grounding 在局部高分辨率区域输出坐标。
- **工具调用也由 RL 学**：模型不仅要学点哪里，还要学什么时候放大、裁剪哪块区域、如何利用工具结果。这些选择同样有成本，所以 reward 需要同时看定位正确性和工具使用效率。 训练用 GRPO/RFT，把连续空间奖励和工具调用成功结合起来。
- **和纯 grounding 的区别**：纯 grounding 只在给定图像上输出点；GUI-Eyes 把视觉输入本身变成可操作对象。 这条路线适合高分辨率专业软件和移动小控件，因为一次性整图推理很容易丢细节。

- **AI 怎么看（状态输入 State）**：全屏截图 + 可主动裁剪/缩放的视觉工具观察。

- **AI 怎么想（决策中心 Controller）**：GRPO/RLVR 类方法：同一任务或状态采样多条候选输出，用组内 reward 标准化得到相对优势，不必单独训练 critic。 这里要抓住一个关键点：GUI agent 的 reward 往往延迟且稀疏，所以论文一定要用某种办法把“最后成功/失败”拆成更可学的信号，例如组内相对优势、step-level progress、成功轨迹 replay、world-model lookahead、任务难度 curriculum 或 chosen/rejected 偏好对。

- **AI 怎么动（动作空间 Action）**：coarse 阶段决定 crop/zoom/重看，fine 阶段输出 grounding 坐标。

- **Reward / 训练信号怎么给**：连续空间定位奖励，结合位置接近度和区域重叠，并奖励有效工具调用。

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
- **基准测试 (Benchmarks)**：ScreenSpot-Pro 等 grounding benchmark。

- **对比基线 (Baselines)**：通常包括同尺寸 SFT/BC 模型、prompting agent、闭源 VLM/LLM、已有 GUI grounding/agent 模型，以及去掉核心模块的 ablation 版本。具体到本文，比较重点是证明 `GUI-Eyes` 这一路线相对纯 SFT、纯 prompting 或朴素 RL/RFT 的收益。

- **核心量化指标**：GUI-Eyes-3B 只用 3K labeled samples，在 ScreenSpot-Pro 达 44.8% grounding accuracy，显著超过 SFT 和 RL baseline。它是“感知动作也可 RL 学习”的代表。

| 项目 | 论文口径 | 读法 |
|---|---|---|
| 主指标 | 任务成功率、grounding accuracy、平均 reward、轨迹成本或系统吞吐 | 不同 benchmark 口径不能直接横比，必须看环境和动作空间。 |
| 训练信号 | 连续空间定位奖励，结合位置接近度和区域重叠，并奖励有效工具调用。 | 判断它是终局 reward、过程 reward、偏好信号还是世界模型反馈。 |
| 环境 | ScreenSpot-Pro 等 grounding benchmark。 | 看是否真实交互、可重置、是否依赖 DOM/HTML/ADB/VM/browser sandbox。 |
| 模型/基线 | SFT、BC、prompting、闭源模型、同尺寸开源模型和消融项 | 重点看是否公平使用同一 backbone、同一任务和同一 step budget。 |

- **消融实验 (Ablation Studies)**：应重点关注四类消融：去掉 reward/verifier 后是否退化；去掉 replay/curriculum/filtering 后是否训练不稳定；去掉 reasoning/tool/world-model 后是否长任务失败；改变数据规模、候选数、环境并行数或 step budget 后性能如何变化。若论文报告系统吞吐，还要同时看 rollout latency、GPU 利用率、VM/浏览器并行数和每条轨迹成本。

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/gui_eyes_arch.png)

![训练流程图](./images/gui_eyes_training.png)

![Reward 或数据流示意图](./images/gui_eyes_reward.png)

![实验结果表](./images/gui_eyes_results.png)

## 4. 局限性与未来方向 (Limitations)
- 这类方法的结论通常强依赖 benchmark 和环境接口。Web/Android/Desktop/OSWorld/ScreenSpot 的状态、动作和 reward 都不同，不能把一个表里的成功率直接推广到所有 GUI 任务。
- 如果 reward 来自 VLM/ORM judge，误判会污染训练；如果 reward 来自规则 verifier，则任务覆盖常受限于可写规则的场景。
- 在线 RL 和世界模型路线都需要大量系统工程：可重置环境、并行 rollout、稳定截图/DOM/ADB/VM、replay buffer、失败恢复和日志审计。
- 对 grounding 论文而言，单步坐标准确率不等于完整任务成功率；对 computer-use 论文而言，允许 GUI-SDK/API/tool call 后，和纯 GUI-only agent 的比较要分开看。
- 未来方向是更可靠的过程奖励、更统一的跨平台 action schema、更低成本的环境/世界模型、更强的错误恢复，以及把学习到的技能长期保存为可复用记忆或工具。
