# Digi-Q: Learning Q-Value Functions for Training Device-Control Agents (2025 - Hao Bai, Yifei Zhou, Li Erran Li, Sergey Levine, Aviral Kumar)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：Digi-Q 就是在不让 AI 继续真实试错的情况下，给它训练一个“动作打分器”，让它每一步先想出多个手机操作，再选那个最可能让任务最终成功的操作。

- 🛠️**这套系统是怎么运作的？（大白话拆解工作流）**：第一步，先收集旧轨迹。作者用 AutoUI checkpoint 在 Android emulator 上跑 AitW 任务，得到一批离线轨迹；每条轨迹里有当前截图、执行动作、下一张截图、最后成功/失败。训练 Digi-Q 时不再进 emulator 试新动作，所以它是 offline RL。第二步，先教 VLM 关注“可操作变化”。普通 VLM 可能看得懂页面上有搜索框、按钮、商品图，但不一定知道“点这里会不会真的跳转”。所以 Digi-Q 用当前截图、动作、下一截图构造一个小任务：这个动作是否让画面发生明显变化？如果变化大就标 yes，否则标 no。这样微调后，VLM 的内部特征会更关注按钮、输入框、点击位置这些和操作后果有关的信息。第三步，冻结 VLM，只训练两个很小的 MLP：一个 Q head 看“状态 + 候选动作”，给这个动作打长期成功分；一个 V head 只看“状态”，估计当前局面平均有多大希望成功。第四步，策略更新时不直接做危险的 policy gradient，而是先从旧策略采样多个候选动作，用 Q head 排序，只让 actor 学那个最高分且确实比当前状态平均水平更好的动作。

  - **AI 怎么看（状态输入 State）**：它看的主要是 Android 当前截图、任务指令、历史交互日志，以及候选动作文本。注意：Digi-Q 的 Q head 不是只看截图，它必须同时看候选动作。例如同一张页面上，“点搜索框”和“点广告横幅”视觉状态一样，但动作不同，未来结果完全不同。论文主文说 Q head 使用 representation fine-tuned VLM 的 yes/no token embedding 作为 `(s,a)` 表征；附录里实际结构更具体：Q features 会拼接 BERT 的动作/文本 embedding、BLIP-2 的视觉 embedding，以及 VLM intermediate-layer representations，再送进 Q MLP。V head 不看动作，实践中用状态视觉特征就够了。

  - **AI 怎么想（决策中心 Controller）**：Digi-Q 的核心 controller 是一个离线训练出来的价值评估器。它不说“专家在这张图上点哪里”，而是回答“如果现在做这个动作，未来完成任务的概率会不会更高”。Q head 的问题是：`当前状态 s + 候选动作 a` 值多少分？V head 的问题是：`当前状态 s` 平均值多少分？两者相减就是 advantage：如果 `Q(s,a) - V(s) > 0`，说明这个动作比当前状态下的平均动作更值得学。训练 Q/V 时用的是 TD 思想：最后成功的奖励会一层层往前传，最终让“打开网站”“点搜索框”“输入商品名”这些早期动作也获得正价值。

  - **AI 怎么动（动作空间 Action）**：它的动作仍是手机 GUI 原子操作：点击归一化坐标、输入文本、滑动、返回等。Digi-Q 本身不扩展动作空间；它的贡献是给这些候选动作排序。推理时流程像这样：actor 先提出 `N` 个动作候选，论文主实验用 `N=16`；Q head 分别打分；系统执行分最高的动作。如果 actor 采样不出正确候选，Q head 也救不了，所以它是 reranker，不是万能规划器。

  - **训练流程用人话写成伪代码**：

```text
1. 用旧 AutoUI 策略收集离线 Android 轨迹。
2. 对每个动作判断：执行后截图有没有明显变化？
3. 用这个 yes/no 小任务微调 VLM，让特征学会关注可操作区域。
4. 冻结 VLM 主体。
5. 训练 Q head：看状态和动作，预测这个动作未来能不能导向成功。
6. 训练 V head：只看状态，预测当前局面平均有多大希望成功。
7. 为每个状态采样多个候选动作。
8. 用 Q 排序，只模仿最高分且 advantage 为正的动作。
```

- 🎯**它解决了前人什么头疼的问题？**：以前有两条路都不舒服。第一条是 DigiRL 这种在线 RL，效果强，但要真实和手机环境交互，成本高、慢、还有风险；真实网站会变、账号会被拦、captcha 会跳出来。第二条是纯 BC/Filtered BC，安全便宜，但只会模仿数据里那一个动作，不知道同一状态下别的候选动作是不是更好。Digi-Q 的妙处是把“在线试错”换成“离线动作打分”：不用真的执行 16 个候选动作，只要让 Q head 预测它们的长期价值，就能在训练和推理时做 Best-of-N 选择。它还用 representation fine-tuning 解决了一个细节痛点：普通 VLM 的特征可能只会描述画面，不会理解“这个点击会不会改变页面”，所以直接接 Q head 容易退化成只看状态、不看动作的 V 函数。

## 2. 实验设置与结果 (Experiments)
- **基准测试 (Benchmarks)**：Digi-Q 评估在 Android-in-the-Wild (AitW) 的 General 和 Web Shopping 子集。每个 episode 从 Android emulator home screen 开始，任务是自然语言指令，agent 通过 pixel-level command 控制设备。论文主文报告离线数据来自预训练 AutoUI checkpoint 的历史 rollouts，Web Shopping 使用 1296 条轨迹，General 使用约 1008 条轨迹；附录还描述了为 Best-of-N 预采样候选动作的工程流程。Web Shopping horizon 为 20，General horizon 为 10。因为训练阶段只用静态 replay buffer，不继续进入 emulator 采样新动作，所以 Digi-Q 是 offline RL / off-policy value learning。

- **对比基线 (Baselines)**：Prompting 组包括 Set-of-Marks + GPT-4V/Gemini 1.5 Pro 和 AppAgent + GPT-4V/Gemini 1.5 Pro。监督组包括 CogAgent 和 AutoUI。学习组包括 Filtered BC、DigiRL offline、Digi-Q，以及允许在线交互的 DigiRL online。Digi-Q 特别对比了三类核心问题：不用 representation fine-tuning 的 off-the-shelf VLM 是否足够；MC return regression 是否能替代 TD learning；REINFORCE/AWR 是否能替代 Best-of-N policy extraction。

- **核心量化指标**：主指标是 AitW task success rate，由 autonomous evaluator 判断。表中是论文 Table 1。

| 方法类别 | 方法 | AitW General Train | AitW General Test | AitW Web Shopping Train | AitW Web Shopping Test |
|---|---|---:|---:|---:|---:|
| Prompting | Set-of-Marks + GPT-4V | 5.2 | 13.5 | 3.1 | 8.3 |
| Prompting | Set-of-Marks + Gemini 1.5 Pro | 32.3 | 16.7 | 6.3 | 11.5 |
| Prompting | AppAgent + GPT-4V | 13.5 | 17.7 | 12.5 | 8.3 |
| Prompting | AppAgent + Gemini 1.5 Pro | 14.6 | 16.7 | 5.2 | 8.3 |
| Supervised Training | CogAgent | 25.0 | 25.0 | 31.3 | 38.5 |
| Supervised Training | AutoUI | 27.7 | 22.9 | 20.7 | 25.0 |
| Learning Offline | Filtered BC | 51.0 +/- 0.9 | 54.5 +/- 1.3 | 37.2 +/- 4.7 | 43.8 +/- 1.7 |
| Learning Offline | DigiRL | 53.5 +/- 2.7 | 59.0 +/- 4.7 | 43.1 +/- 3.6 | 47.6 +/- 4.2 |
| Learning Offline | Digi-Q | 61.5 +/- 2.3 | 71.2 +/- 2.1 | 53.1 +/- 1.7 | 58.0 +/- 2.1 |
| Learning Online | DigiRL | 63.5 +/- 3.1 | 74.5 +/- 2.6 | 52.6 +/- 1.6 | 57.3 +/- 3.1 |

Digi-Q 在两个 test 子集上都超过所有离线方法。相对 DigiRL offline，论文报告平均约 21.2% relative improvement；相对 Filtered BC，平均约 31.5% relative improvement。更重要的是，Digi-Q 在 Web Shopping Test 上 58.0%，已经接近甚至略高于允许在线交互的 DigiRL 57.3%，说明一个可靠 Q 函数可以用离线数据实现强策略改进。

- **消融实验 (Ablation Studies)**：representation fine-tuning 是最大关键之一。Web Shopping Test 上，Behavior Policy 只有 25.0；直接用 off-the-shelf VLM 表征训练 Q 只有 31.9 +/- 1.3，说明普通 VLM 虽然看得懂画面，但不一定知道“动作会导致什么状态变化”；用 BLIP-2 + BERT 表征达到 47.6 +/- 5.2；完整 Digi-Q 达到 58.0 +/- 2.1。MC return regression 只有 37.5 +/- 4.5，比完整 TD learning 低约 20 个点，说明 TD backup 的 credit assignment 比整条轨迹回报回归更有效。

| 表征/critic 方案 | Web Shopping Test Success |
|---|---:|
| Behavior Policy | 25.0 |
| Digi-Q w/ MC return | 37.5 +/- 4.5 |
| Digi-Q Off-the-shelf VLM | 31.9 +/- 1.3 |
| Digi-Q w/ BLIP-2 + BERT | 47.6 +/- 5.2 |
| Digi-Q 完整方法 | 58.0 +/- 2.1 |

actor objective 消融显示，Best-of-N 比 REINFORCE 和 AWR 更稳定也更强。REINFORCE 有提升但 KL 很大、方差高；AWR KL 小但太保守，甚至低于 behavior policy；Digi-Q 在中等 KL 下达到最好性能。

| Actor Objective | Web Shopping Test Success | Token-level KL vs Behavior |
|---|---:|---:|
| Behavior Policy | 25.0 | 0 |
| REINFORCE | 37.5 +/- 4.7 | 7.15 |
| AWR | 19.4 +/- 1.3 | 2.84 |
| Digi-Q Best-of-N | 58.0 +/- 2.1 | 3.28 |

Best-of-N 的 `N` 也做了消融，`N in {1,4,8,16}` 时性能随 `N` 增大单调提升，支持“Q 函数可以在离线状态下比较更多候选动作”的假设。数据效率实验显示，在 256 条轨迹的低数据区间，Digi-Q 也优于 DigiRL offline，说明 Q/V 的 per-step credit assignment 比只过滤成功轨迹更省数据。定性图还显示完整 Digi-Q 的 advantage `Q(s,a)-V(s)` 和人类对动作好坏的判断更一致，而 MC、off-the-shelf VLM 或无 representation fine-tuning 的 Q 函数容易退化成几乎不看动作的 state-only value。

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/digiq_arch.png)

![Representation fine-tuning 示意图](./images/digiq_representation_tuning.png)

![Q/V head 网络结构图](./images/digiq_q_value_head.png)

![Best-of-N policy extraction 图](./images/digiq_best_of_n.png)

![消融实验表格截图](./images/digiq_ablation.png)

## 4. 局限性与未来方向 (Limitations)
- Digi-Q 的能力受离线轨迹覆盖限制。它可以在已有状态上比较多个候选动作，但如果离线数据没有覆盖某类 app 状态、弹窗、登录流或错误恢复情形，Q 函数对这些 OOD 状态的评分就不可靠。
- Best-of-N 依赖候选生成质量。如果 actor 根本采样不到正确动作，Q 函数只能在坏候选中挑相对好的动作，不能凭空发现候选集合之外的操作。
- Reward 和 evaluation 仍依赖 proprietary VLM evaluator。即使训练不在线交互，离线轨迹中的 reward 标签和测试成功率仍由 evaluator 决定，存在误判和 reward hacking 风险。
- 方法主要验证在 Android/AitW 设备控制上。论文认为方法一般化，但没有证明可直接用于 WindowsAgentArena、OSWorld、VSCode 编程、Office 文档编辑等更复杂 GUI/OS 任务。
- Q/V 使用冻结 VLM 表征提升稳定性，但当 policy 分布持续漂移时，critic 可能需要 active online self-improvement；作者也指出如果把 Digi-Q critic 放进在线闭环，需要更复杂的系统设计和更鲁棒的 critic。
