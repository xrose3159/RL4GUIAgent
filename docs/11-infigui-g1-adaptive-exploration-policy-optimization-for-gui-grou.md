# InfiGUI-G1: Adaptive Exploration Policy Optimization for GUI Grounding (2025 - Yuhang Liu, Zeyu Liu, Shuanghe Zhu et al.)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：提出 AEPO。通过多答案生成扩大探索，再用 Adaptive Exploration Reward 从效率 eta=U/C 出发引导探索，把有用且成本低的探索分配更高权重。它还强调 grounding 中强制 CoT 可能伤害坐标精度。

- **这套系统是怎么运作的？（大白话工作流）**：RLVR 能改善空间对齐，但对语义对齐不足，尤其是多个相似 UI 元素中要选功能正确的元素时，探索效率成为瓶颈。 论文的做法可以理解成把 GUI agent 拆成“看状态、产动作、拿反馈、再更新”的闭环。与普通 SFT 最大的区别是，它不只学习专家下一步，而是把环境反馈、规则验证、reward model、搜索结果或世界模型预测变成可训练信号。

- **问题：坐标任务也需要探索**：在一张 GUI 截图里，经常有多个相似按钮、重复列表项或同名控件。模型只生成一个点时，很容易卡在外观相似但语义错误的元素上。 InfiGUI-G1 让模型生成多个候选答案，再用 reward 区分哪些探索有用，核心是提升 grounding 中的语义探索效率。
- **AEPO：用 useful/cost 衡量探索效率**：AEPO 不鼓励无限长 reasoning 或大量重复候选。它定义探索效率 η = U/C，U 表示候选带来的有效信息或正确性提升，C 表示生成成本、冗余和无效探索。 高效候选得到更高权重；冗长但没定位到目标的 CoT 会被压低。这就是它为什么强调强制长 CoT 可能伤害坐标任务。
- **最终学到什么**：模型要学的不只是“这个像按钮”，而是“哪个按钮满足当前任务语义”。比如多个 Add 按钮里，要根据列表项文字、位置关系和任务目标选择正确一项。 所以 InfiGUI-G1 介于纯 grounding 和语义 reasoning 之间，目标是让坐标预测不只依赖外观相似性。

- **AI 怎么看（状态输入 State）**：GUI 截图、元素指令和多个候选答案。

- **AI 怎么想（决策中心 Controller）**：混合训练流程：通常先 SFT/数据冷启动，再用规则 reward、VLM judge、process reward 或 replay/filtering 做 RFT/RL/自演化。 这里要抓住一个关键点：GUI agent 的 reward 往往延迟且稀疏，所以论文一定要用某种办法把“最后成功/失败”拆成更可学的信号，例如组内相对优势、step-level progress、成功轨迹 replay、world-model lookahead、任务难度 curriculum 或 chosen/rejected 偏好对。

- **AI 怎么动（动作空间 Action）**：生成多个 grounding 候选点/答案。

- **Reward / 训练信号怎么给**：Adaptive Exploration Reward，按 useful exploration / exploration cost 给权重，避免高成本无效探索。

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
- **基准测试 (Benchmarks)**：多项 GUI grounding benchmark，重点看泛化和语义对齐。

- **对比基线 (Baselines)**：通常包括同尺寸 SFT/BC 模型、prompting agent、闭源 VLM/LLM、已有 GUI grounding/agent 模型，以及去掉核心模块的 ablation 版本。具体到本文，比较重点是证明 `InfiGUI-G1` 这一路线相对纯 SFT、纯 prompting 或朴素 RL/RFT 的收益。

- **核心量化指标**：InfiGUI-G1-3B/7B 在多项 GUI grounding benchmark 上达到新 SOTA，相比 naive RLVR 在考查泛化和语义理解的 benchmark 上最高相对提升 9.0%，并被 AAAI 2026 接收为 Oral。

| 项目 | 论文口径 | 读法 |
|---|---|---|
| 主指标 | 任务成功率、grounding accuracy、平均 reward、轨迹成本或系统吞吐 | 不同 benchmark 口径不能直接横比，必须看环境和动作空间。 |
| 训练信号 | Adaptive Exploration Reward，按 useful exploration / exploration cost 给权重，避免高成本无效探索。 | 判断它是终局 reward、过程 reward、偏好信号还是世界模型反馈。 |
| 环境 | 多项 GUI grounding benchmark，重点看泛化和语义对齐。 | 看是否真实交互、可重置、是否依赖 DOM/HTML/ADB/VM/browser sandbox。 |
| 模型/基线 | SFT、BC、prompting、闭源模型、同尺寸开源模型和消融项 | 重点看是否公平使用同一 backbone、同一任务和同一 step budget。 |

- **消融实验 (Ablation Studies)**：应重点关注四类消融：去掉 reward/verifier 后是否退化；去掉 replay/curriculum/filtering 后是否训练不稳定；去掉 reasoning/tool/world-model 后是否长任务失败；改变数据规模、候选数、环境并行数或 step budget 后性能如何变化。若论文报告系统吞吐，还要同时看 rollout latency、GPU 利用率、VM/浏览器并行数和每条轨迹成本。

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/infigui_g1_arch.png)

![训练流程图](./images/infigui_g1_training.png)

![Reward 或数据流示意图](./images/infigui_g1_reward.png)

![实验结果表](./images/infigui_g1_results.png)

## 4. 局限性与未来方向 (Limitations)
- 这类方法的结论通常强依赖 benchmark 和环境接口。Web/Android/Desktop/OSWorld/ScreenSpot 的状态、动作和 reward 都不同，不能把一个表里的成功率直接推广到所有 GUI 任务。
- 如果 reward 来自 VLM/ORM judge，误判会污染训练；如果 reward 来自规则 verifier，则任务覆盖常受限于可写规则的场景。
- 在线 RL 和世界模型路线都需要大量系统工程：可重置环境、并行 rollout、稳定截图/DOM/ADB/VM、replay buffer、失败恢复和日志审计。
- 对 grounding 论文而言，单步坐标准确率不等于完整任务成功率；对 computer-use 论文而言，允许 GUI-SDK/API/tool call 后，和纯 GUI-only agent 的比较要分开看。
- 未来方向是更可靠的过程奖励、更统一的跨平台 action schema、更低成本的环境/世界模型、更强的错误恢复，以及把学习到的技能长期保存为可复用记忆或工具。
