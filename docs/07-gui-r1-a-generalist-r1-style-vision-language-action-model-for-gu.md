# GUI-R1: A Generalist R1-Style Vision-Language Action Model for GUI Agents (2025 - Run Luo, Lu Wang, Wanwei He et al.)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：统一 action space，把 click、swipe、type 等动作编码为可规则验证的格式；用少量高质量跨平台数据做 GRPO，奖励包括动作格式、动作类型和坐标是否落在目标框内。

- **这套系统是怎么运作的？（大白话工作流）**：大多数 GUI agent 靠 SFT，需要大量数据，而且对未见界面泛化差。DeepSeek-R1 式 RLVR 启发作者用规则奖励提升 GUI action model。 论文的做法可以理解成把 GUI agent 拆成“看状态、产动作、拿反馈、再更新”的闭环。与普通 SFT 最大的区别是，它不只学习专家下一步，而是把环境反馈、规则验证、reward model、搜索结果或世界模型预测变成可训练信号。

- **任务不是在线执行，而是静态下一步 action prediction**：GUI-R1 的输入是高层任务 Q、当前截图 I 和历史 H，输出下一步动作：action type、click point 和 input text。训练时它不会真的点击环境，也不会得到新截图；它是在已有 benchmark/标注样本上做 rule-based reinforcement fine-tuning。 因此它的“RL”是 non-interactive RFT：模型对同一输入生成多个候选回答，规则 reward 和 ground truth 比较后打分，GRPO 根据组内相对分数更新策略。
- **统一动作空间：把跨平台动作压成可检查格式**：GUI-R1 把动作统一成 complete、close/delete、press home、click、press back、type、select、scroll、enter。最终答案固定为 action、point、input text 三个字段，并要求模型用 <think> 和 <answer> 标签输出。 这个统一格式解决了 Android、Web、Desktop 数据动作定义不一致的问题，也让 reward 可以写成简单规则：动作类型是否相等、点击点是否落在 bbox 内、输入文本是否匹配。
- **GRPO 训练：不训 critic，用组内相对奖励**：对每个样本，模型生成 N 个候选回答 o_i，每个候选得到 reward r_i。优势不是 Q-V，也不是 GAE，而是 A_i = (r_i - mean(r_1...r_N)) / std(r_1...r_N)。高于组平均的回答概率上升，低于组平均的回答概率下降。 reward 分为格式奖励 R_f 和准确性奖励 R_acc，最终 R_o = αR_f + βR_acc。R_acc = R_act + R_point + R_text：动作类型 exact match，点击坐标落入 GT bbox，文本 F1 超过阈值。
- **GUI-R1-3K：保留中等难度样本**：作者先从约 14M grounding/low-level 数据和约 30K high-level 数据出发，用 Qwen2.5-VL-7B 对每个样本生成 10 个回答，再用规则 reward 打分。全 0 的样本太难或标注不清，全 1 的样本太简单，都被过滤。 最后保留约 1.5K high-level 和随机采样 1.5K low-level，组成 3K 训练集。GUI-R1-3B 是从 Qwen2.5-VL-3B-Instruct 直接做 RFT/GRPO，不是先在 GUI-R1-3K 上 SFT 再 RL；带星号的 QwenVL2.5-3B* 才是同数据 SFT baseline。

- **AI 怎么看（状态输入 State）**：截图 + 高层任务指令 + 历史动作。它是静态 action prediction/RFT，不进入真实环境执行动作；训练时同一输入采样多个动作候选。

- **AI 怎么想（决策中心 Controller）**：GRPO/RLVR 类方法：同一任务或状态采样多条候选输出，用组内 reward 标准化得到相对优势，不必单独训练 critic。 这里要抓住一个关键点：GUI agent 的 reward 往往延迟且稀疏，所以论文一定要用某种办法把“最后成功/失败”拆成更可学的信号，例如组内相对优势、step-level progress、成功轨迹 replay、world-model lookahead、任务难度 curriculum 或 chosen/rejected 偏好对。

- **AI 怎么动（动作空间 Action）**：统一为 `complete/close/delete/home/click/back/type/select/scroll/enter` 等，并输出 action、point、input text。坐标用归一化点，click 类动作用 bbox 命中判断。

- **Reward / 训练信号怎么给**：规则奖励：格式奖励 + 动作类型奖励 + 坐标命中奖励 + 文本匹配奖励。GRPO 用同组候选的相对 reward 更新，不训练 critic。

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
- **基准测试 (Benchmarks)**：ScreenSpot、ScreenSpot-v2、ScreenSpot-Pro、AndroidControl、AITW/GUI action prediction 等 mobile/desktop/web benchmark。

- **对比基线 (Baselines)**：通常包括同尺寸 SFT/BC 模型、prompting agent、闭源 VLM/LLM、已有 GUI grounding/agent 模型，以及去掉核心模块的 ablation 版本。具体到本文，比较重点是证明 `GUI-R1` 这一路线相对纯 SFT、纯 prompting 或朴素 RL/RFT 的收益。

- **核心量化指标**：只用 3K 样本，即 OS-Atlas 13M 数据的 0.02%，在 mobile/desktop/web 三个平台的 8 个 benchmark 上达到或超过 SOTA。综述特别强调其出现了未显式监督的“先观察布局、再定位元素”的 System-2 风格内省。

| 项目 | 论文口径 | 读法 |
|---|---|---|
| 主指标 | 任务成功率、grounding accuracy、平均 reward、轨迹成本或系统吞吐 | 不同 benchmark 口径不能直接横比，必须看环境和动作空间。 |
| 训练信号 | 规则奖励：格式奖励 + 动作类型奖励 + 坐标命中奖励 + 文本匹配奖励。GRPO 用同组候选的相对 reward 更新，不训练 critic。 | 判断它是终局 reward、过程 reward、偏好信号还是世界模型反馈。 |
| 环境 | ScreenSpot、ScreenSpot-v2、ScreenSpot-Pro、AndroidControl、AITW/GUI action prediction 等 mobile/desktop/web benchmark。 | 看是否真实交互、可重置、是否依赖 DOM/HTML/ADB/VM/browser sandbox。 |
| 模型/基线 | SFT、BC、prompting、闭源模型、同尺寸开源模型和消融项 | 重点看是否公平使用同一 backbone、同一任务和同一 step budget。 |

- **消融实验 (Ablation Studies)**：应重点关注四类消融：去掉 reward/verifier 后是否退化；去掉 replay/curriculum/filtering 后是否训练不稳定；去掉 reasoning/tool/world-model 后是否长任务失败；改变数据规模、候选数、环境并行数或 step budget 后性能如何变化。若论文报告系统吞吐，还要同时看 rollout latency、GPU 利用率、VM/浏览器并行数和每条轨迹成本。

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/gui_r1_arch.png)

![训练流程图](./images/gui_r1_training.png)

![Reward 或数据流示意图](./images/gui_r1_reward.png)

![实验结果表](./images/gui_r1_results.png)

## 4. 局限性与未来方向 (Limitations)
- 这类方法的结论通常强依赖 benchmark 和环境接口。Web/Android/Desktop/OSWorld/ScreenSpot 的状态、动作和 reward 都不同，不能把一个表里的成功率直接推广到所有 GUI 任务。
- 如果 reward 来自 VLM/ORM judge，误判会污染训练；如果 reward 来自规则 verifier，则任务覆盖常受限于可写规则的场景。
- 在线 RL 和世界模型路线都需要大量系统工程：可重置环境、并行 rollout、稳定截图/DOM/ADB/VM、replay buffer、失败恢复和日志审计。
- 对 grounding 论文而言，单步坐标准确率不等于完整任务成功率；对 computer-use 论文而言，允许 GUI-SDK/API/tool call 后，和纯 GUI-only agent 的比较要分开看。
- 未来方向是更可靠的过程奖励、更统一的跨平台 action schema、更低成本的环境/世界模型、更强的错误恢复，以及把学习到的技能长期保存为可复用记忆或工具。
