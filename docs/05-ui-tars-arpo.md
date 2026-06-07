# UI-TARS / ARPO: Native GUI Agents and Agentic Replay Policy Optimization (2025 - UI-TARS Team; Xinyu Chen et al.)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：UI-TARS 先把 GUI 操作统一成“看截图、先思考、再输出动作”的原生 VLM agent，ARPO 再在这个底座上解决 GUI RL 最痛的“整批 rollout 全失败，GRPO 没梯度”问题：把历史成功轨迹按任务缓存起来，当前 batch 全失败时拿成功轨迹补进训练组。

- **这篇文档为什么把 UI-TARS 和 ARPO 放一起**：UI-TARS 论文主要贡献是 native GUI agent 底座：截图输入、统一动作空间、thought-before-action、perception/grounding/action trace 大数据、reflection tuning 和 DPO。它本身不是完整在线 RL 论文。ARPO 论文则明确基于 UI-Tars-1.5 / Qwen2.5-VL 做 end-to-end GUI policy optimization，用 GRPO + experience replay 在 OSWorld 上训练。因此理解 ARPO 必须先理解 UI-TARS 的状态、动作、模型输出格式和训练底座。

- **UI-TARS 底座怎么运作**：

```text
任务 instruction
        ↓
最近 N 张 screenshot + 历史 thought/action 作为短期记忆
        ↓
Qwen-2-VL / UI-TARS VLM
        ↓
先生成 thought：解释当前界面、目标和下一步意图
        ↓
再生成 action：Click / Drag / Type / Scroll / Hotkey / PressBack / Finished ...
        ↓
真实设备或虚拟环境执行动作，返回下一张 screenshot
        ↓
循环直到 Finished、CallUser、失败或达到步数上限
```

  UI-TARS 的核心建模式子是：

$$
P(t_n,a_n \mid instruction, t_1,a_1,\ldots,(o_{n-i},t_{n-i},a_{n-i})_{i=1}^{N}, o_n)
$$

  其中 $o_n$ 是当前 screenshot，$t_n$ 是 thought，$a_n$ 是 action。它不是先让一个 OCR 模块读屏、一个 planner 做计划、一个 grounding 模块点坐标，而是把这些能力尽量收进一个原生 VLM policy 里。

- **AI 怎么看（State）**：UI-TARS 只依赖 screenshot 作为主观察，不要求 DOM 或 accessibility tree。训练时会用解析工具从网站、app、操作系统界面抽取 element type、bounding box、text、depth 等 metadata 来构造监督数据，但推理时模型面对的是截图和历史上下文。为了让它看懂 GUI，作者构造了五类 perception 数据：element description、dense captioning、state transition captioning、GUI QA、Set-of-Mark 相关训练。这里最重要的是 state transition captioning：给模型两张连续截图，让它描述界面发生了什么变化，这为后面多步 agent 判断“刚才操作是否有效”打基础。

- **AI 怎么动（Action）**：UI-TARS 定义跨平台统一动作空间，把 Windows 的 click、Android 的 tap、网页点击都规范成统一语义。共享动作包括：

| 动作 | 含义 |
|---|---|
| `Click(x, y)` | 点击归一化坐标 |
| `Drag(x1, y1, x2, y2)` | 从起点拖到终点 |
| `Scroll(x, y, direction)` | 在某点附近滚动 |
| `Type(content)` | 输入文本 |
| `Wait()` | 等待页面/动画/加载 |
| `Finished()` | 声明任务完成 |
| `CallUser()` | 需要登录、验证码、隐私信息等人工介入 |

  桌面额外有 `Hotkey(key)`、`LeftDouble(x,y)`、`RightSingle(x,y)`；移动端额外有 `LongPress(x,y)`、`PressBack()`、`PressHome()`、`PressEnter()`。Grounding 训练时，模型要把元素描述映射到归一化坐标中心点，所以它不是靠 SoM 编号强行点按钮，而是学会直接预测坐标。

- **UI-TARS 怎么训练**：训练分三段。第一段是 continual pre-training，使用约 50B tokens 的 GUI 相关数据，覆盖 perception、grounding、action traces，不含 reflection tuning。第二段是 annealing phase，选更高质量的 perception、grounding、action trace 和 reflection 数据继续训练，得到 UI-TARS-SFT。第三段是 DPO phase，用 online bootstrapping 中产生的错误动作和人工修正动作构造偏好对，得到 UI-TARS-DPO。

  Reflection tuning 的关键不是只告诉模型“正确动作是什么”，而是专门让模型看到自己犯错后的恢复路径。假设第 $\tau$ 步动作错了，标注者给出修正 thought/action：

```text
T-: (..., o_tau, t_tau,   a_tau)      # 原错误动作
T+: (..., o_tau, t*_tau, a*_tau)      # 修正动作
```

  还会标注“错已经发生之后下一步怎么补救”：

```text
T-: (..., o_tau,t_tau,a_tau, o_{tau+1},t_{tau+1},a_{tau+1})
T+: (..., o_tau,t_tau,a_tau, o_{tau+1},t*_{tau+1},a*_{tau+1})
```

  SFT 只在修正步骤上算 loss；DPO 则显式把 corrected action 当 chosen、erroneous action 当 rejected：

$$
L_{DPO}(\theta)=
-\mathbb{E}
\log\sigma\left(
\beta\log\frac{\pi_\theta(a'_\tau|s_\tau)}{\pi_{SFT}(a'_\tau|s_\tau)}
-
\beta\log\frac{\pi_\theta(a_\tau|s_\tau)}{\pi_{SFT}(a_\tau|s_\tau)}
\right)
$$

  大白话：同一个界面历史下，增加修正动作概率，降低原错误动作概率，而且别偏离 SFT 模型太远。

- **ARPO 的 RL 核心怎么做**：ARPO 把 GUI 任务建成 multi-turn MDP。每条 trajectory 是多张 1080P screenshot 和鼠标/键盘动作序列，最后环境给一个 scalar reward。模型基于 UI-Tars / Qwen2.5-VL，输入不是最近一两帧，而是尽量利用完整 screenshot-action history 来推理长程依赖。

  ARPO 用的是 GRPO。对同一个任务 query $q$，采样 $G$ 条 rollout，每条得到总 reward $r_i$。GRPO 不训练 value function，而是用组内 reward 均值和标准差做 advantage：

$$
\hat{A}_{i,t}=\frac{r_i-\mu}{\sigma}
$$

  然后用 PPO-style clipped objective 更新 token probability。它的好处是省掉 critic，适合长上下文、多图 VLM；问题是如果同一个任务的 $G$ 条 rollout 全部失败，所有 $r_i=0$，组内方差 $\sigma=0$，advantage 全是 0，训练等于没有梯度。

- **ARPO 的 replay buffer 为什么关键**：ARPO 的解决办法是 per-task successful trajectory replay buffer。每个任务只缓存成功轨迹。如果某个 GRPO group 当前采样的 rollout 全部 reward=0，就从该任务 replay buffer 中随机取一条历史成功轨迹，替换组里一条失败轨迹。这样组里至少有一个正 reward，reward variance 不为 0，token-level advantage 能算出来。

```text
for task x:
    rollouts = sample G trajectories with current policy
    rewards = OSWorld_evaluator(rollouts)

    if all rewards == 0 and replay_buffer[x] has success:
        rollouts[0] = sample_success(replay_buffer[x])
        rewards[0] = positive_reward

    update policy with GRPO(rollouts, rewards)

    for successful rollout:
        replay_buffer[x].add(rollout)
```

  这不是普通离线 replay。它只在“当前组全失败导致 GRPO 没信号”时救场，目的是维持组内 reward diversity。buffer 固定大小，满了淘汰最旧轨迹，避免离当前 policy 太远。

- **Reward 设计**：ARPO 的 reward 有两部分。第一是 trajectory reward：OSWorld 环境根据任务完成情况给 $r_t=1$ 或 $0$。第二是 action format reward：如果 VLM 输出不能被 parser 解析成合法 GUI action，就给 $r_f=-1$。训练目标可以写成：

$$
\max_\theta\ \mathbb{E}_{x\sim D,\tau\sim\pi_\theta}
\left[r_t(x,\tau)+r_f(x,\tau)\right]
$$

- **Task selection 怎么做**：ARPO 不是拿 OSWorld 全部任务一股脑训练。作者先用 UI-Tars-1.5 对每个 OSWorld 任务跑 16 次；只要至少成功一次，就保留为有学习信号的 valuable task。这样筛出 128 个任务用于 GRPO/ARPO 训练。原因很简单：完全做不了的任务只会产生全 0 batch；已经太简单的任务提升有限；最适合训练的是能偶尔成功、但还不稳定的任务。

- 🎯**它解决了前人什么头疼的问题**：UI-TARS 解决的是“GUI agent 不能只靠模块拼装和 DOM/规则”的问题，把 perception、grounding、thought、action 统一进一个 screenshot-to-action 模型。ARPO 解决的是“把 GRPO 直接搬到 GUI 会失效”的问题：GUI 任务长、慢、稀疏奖励，早期整批失败很常见；ARPO 用 valuable task selection 先保证任务有成功可能，再用 success replay 保证组内有 reward variance，让 GRPO 在真实桌面环境里终于能持续优化。

## 2. 实验设置与结果 (Experiments)
- **基准测试 (Benchmarks)**：UI-TARS 评估 perception、grounding 和 agent execution 三类能力。关键在线环境包括 OSWorld 和 AndroidWorld；OSWorld 是 Ubuntu/Windows/macOS 风格真实桌面任务，AndroidWorld 是 live Android emulator 任务。ARPO 专注 OSWorld，使用执行脚本给 reward；并提出 OSWorld Hard，禁止到步数上限时把最后动作替换成 `FAIL` 来规避 impossible-task reward hacking。ARPO 所有模型评估最大步数限制为 15。

- **对比基线 (Baselines)**：UI-TARS 对比 GPT-4o、Claude Computer Use、Gemini、CogAgent、Aguvis、OS-Atlas、Aria-UI、UGround、ShowUI 等。ARPO 对比 Aria-UI、Aguvis、OS-Atlas、UI-Tars-7B-DPO、UI-Tars-7B-1.5、GRPO-only、ARPO，并额外比较 Reject Sampling、DPO、KTO 等离线偏好优化方法。

- **UI-TARS 在线 benchmark 关键结果**：

| 方法 | OSWorld | AndroidWorld |
|---|---:|---:|
| GPT-4o direct | 5.0 | 34.5 (SoM) |
| Gemini-Pro-1.5 | 5.4 | 22.8 (SoM) |
| CogAgent-9B | 8.1 | 26.1 |
| Aguvis-72B | 10.3 | 27.9 |
| Claude Computer Use | 14.9 (15 steps) / 22.0 (50 steps) | - |
| UI-TARS-7B-SFT | 17.7 (15 steps) | 33.0 |
| UI-TARS-7B-DPO | 18.7 (15 steps) | - |
| UI-TARS-72B-SFT | 18.8 (15 steps) | 46.6 |
| UI-TARS-72B-DPO | 22.7 (15 steps) / 24.6 (50 steps) | - |

  这里要注意：UI-TARS 的 DPO 对 OSWorld 特别有用，因为 OSWorld 更需要错误恢复和长程 reasoning；AndroidWorld 表中主要报告 SFT 模型，72B-SFT 达到 46.6，超过 GPT-4o + SoM 的 34.5。

- **ARPO 主结果**：

| 模型 | OSWorld | OSWorld Hard |
|---|---:|---:|
| Aria-UI + GPT-4o | 15.2 | - |
| Aguvis-7B + GPT-4o | 14.8 | - |
| Aguvis-72B + GPT-4o | 17.0 | - |
| OS-Atlas-7B + GPT-4o | 14.6 | - |
| UI-Tars-7B-DPO | 15.6 | 11.3 |
| UI-Tars-7B-DPO + GRPO | 18.3 | 16.4 |
| UI-Tars-7B-DPO + ARPO | 20.4 | 18.0 |
| UI-Tars-7B-1.5 | 23.5 | 18.2 |
| UI-Tars-7B-1.5 + GRPO | 26.0 | 20.9 |
| UI-Tars-7B-1.5 + ARPO | 29.9 | 23.8 |

  ARPO 在 UI-Tars-7B-1.5 上把 OSWorld 从 23.5 提到 29.9，把 OSWorld Hard 从 18.2 提到 23.8；在 UI-Tars-7B-DPO 上也从 15.6 提到 20.4。

- **消融实验 (Ablation Studies)**：第一，replay buffer 是 ARPO 的核心。带 replay 的模型在训练约 Step 30 后开始稳定超过无 replay，最终平均 trajectory reward 为 0.75 vs 0.65；in-domain task success 从 GRPO 的 68.8% 提到 ARPO 的 81.25%，绝对提升 12.5%。第二，task selection 很关键：用 128 个 valuable tasks 训练，比用完整 OSWorld 任务训练有更高 reward variance 和更快收敛；GRPO 依赖组内 reward 差异，任务全失败会让 advantage 消失。第三，OOD 泛化更难：base UI-Tars-1.5 OOD 为 55.2%，GRPO 为 52.08%，ARPO 为 56.3%，说明 replay 能缓解过拟合，但真正泛化还需要更多任务多样性。第四，直接 trajectory-level RL 比离线偏好优化更强：ARPO 27.3%，GRPO 26.0%，KTO 24.6%，DPO 22.4%，Reject Sampling 21.8%。

## 3. 图文占位符 (Visual Guides)
![UI-TARS 系统架构图](./images/uitars_arch.png)

![统一动作空间表](./images/uitars_action_space.png)

![Reflection tuning 与 DPO 偏好对](./images/uitars_reflection_dpo.png)

![ARPO 训练流程图](./images/arpo_training_flow.png)

![Experience replay buffer 消融图](./images/arpo_replay_ablation.png)

![OSWorld 主结果表](./images/arpo_osworld_results.png)

## 4. 局限性与未来方向 (Limitations)
- UI-TARS 的底座训练依赖非常大的 GUI 数据工程：截图、metadata、action trace、reflection annotation、DPO pairs，总量约 50B tokens；这对普通研究者复现门槛很高。
- UI-TARS 是 screenshot-only native agent，泛化强但也意味着它放弃了 DOM/accessibility tree 这类结构化信息；在需要精确表单结构、网页 hidden state 或复杂软件状态时，纯截图可能不够。
- ARPO 仍然依赖 OSWorld 的规则 reward 和可重置 VM 环境。它证明了 desktop GUI RL 可行，但没有直接覆盖 Android、WebArena、真实浏览器账号场景或跨 app 长任务。
- ARPO 的 replay buffer 只缓存成功轨迹，能解决全 0 batch，却不能细粒度利用失败轨迹里的正确前缀；相比 value/Q 方法，它的 credit assignment 仍然是 trajectory-level。
- Task selection 会带来训练分布偏差：只训练“至少成功一次”的 128 个任务能稳定起步，但可能降低对完全未解决任务和 OOD 任务的覆盖。
- 未来方向是结合 step-level value/reward model、更多样的 task curriculum、更强 verifier，以及把 ARPO 的 replay 思想扩展到 Web、Android、WindowsAgentArena 等更复杂多平台环境。
