# Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents (2024 - Pranav Putta, Edmund Mills, Naman Garg et al.)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：Agent Q 就是把网页操作当成一棵“闯关搜索树”，让 AI 先在树里试出好路和坏路，再用这些成败分支教大模型以后少走弯路。

- 🛠️**这套系统是怎么运作的？（大白话拆解工作流）**：第一步，LLM 先像普通 ReAct agent 一样读任务、看网页、想下一步。例如用户说“帮我在 OpenTable 订某家餐厅 4 人 7 点的位置”，模型先写一个计划，再输出下一步浏览器动作。第二步，不急着只执行一个动作到底，而是在当前网页状态下让模型提出多个可能动作，比如点搜索框、输入餐厅名、滚动页面、点某个时间槽。第三步，用 MCTS 像下棋一样搜索：每个网页状态是树上的一个节点，每个动作是一条边，走到新页面就展开新节点。第四步，因为网页任务通常只有最后成功/失败，没有每一步分数，所以 Agent Q 又让一个 LLM critic 对候选动作做过程评价：哪个动作看起来更有助于完成任务，哪个动作可能走偏。第五步，搜索结束后，把成功分支当正例、失败或低分分支当负例，做 step-level DPO，让策略以后在同样页面历史下更倾向好动作、更少选坏动作。

  - **AI 怎么看（状态输入 State）**：Agent Q 不是只看截图。WebShop 里页面主要用环境提供的文本/DOM 类表示；OpenTable 真实网页里，系统会抓取 raw HTML，抽取相关视觉组件，并给可交互元素标 ID，让模型能说 `CLICK [ID]` 或 `TYPE [ID] [TEXT]`。它还会保留一个紧凑历史 `h_t`：前面做过哪些动作、当前网页是什么样。这样做是因为完整网页 DOM 可能有几十万 token，完整塞给 LLM 不现实。

  - **AI 怎么想（决策中心 Controller）**：决策中心有两层。第一层是 LLM policy，负责提出动作和理由；它像一个网页操作员，会生成 plan、thought、environment action、explanation。第二层是 MCTS + critic，负责审查和搜索。MCTS 不是让模型贪心地只走一步，而是多试几条分支；critic 则给每个候选动作一个过程评分。最后，每个动作的价值大致来自两部分：这条分支最后有没有成功，以及 critic 认为这一步本身是否靠谱。论文里的 `Q_feedback(h,a)` 可以用一句话理解：在当前网页历史 `h` 下做动作 `a`，这条路综合看有多值得走。

  - **AI 怎么动（动作空间 Action）**：Agent Q 的动作是浏览器级动作，不是手机坐标点击。WebShop 中常见动作是点击元素、滚动、输入内容、询问用户等。OpenTable 中动作写得更具体：`CLICK [ID]`、`GOTO [URL]`、`TYPE [ID] [TEXT]`、`SUBMIT [ID]`、`CLEAR [ID]`、`SCROLL [UP/DOWN]`、`ASK USER HELP`。它依赖元素 ID 和网页结构，因此比“纯截图点坐标”更像一个读 HTML 的网页自动化 agent。

  - **训练流程用人话写成伪代码**：

```text
1. 从一个基础 web agent policy 开始。
2. 对一个网页任务，读取当前页面和历史动作。
3. 让 policy 提出 K 个候选动作。
4. 让 critic 给这些候选动作排序，判断哪个更可能推进任务。
5. 用 MCTS 选择值得探索的动作分支。
6. 沿分支真实执行到任务结束，得到成功/失败。
7. 把终局成败从叶子节点回传到路径上的动作。
8. 混合“搜索得到的成功率”和“critic 的过程评分”，得到 Q_feedback。
9. 在同一状态下，把高 Q_feedback 动作当 chosen，低 Q_feedback 动作当 rejected。
10. 用 DPO 更新 LLM policy，让它以后更偏向 chosen、远离 rejected。
```

- 🎯**它解决了前人什么头疼的问题？**：以前的 web agent 很容易陷入“看起来合理但其实短视”的动作。比如 WebShop 里模型搜索商品后只看第一页结果，不会点下一页；OpenTable 里如果目标时间没有直接出现，模型可能不知道改找相近时间或问用户。纯 SFT 只告诉模型专家怎么做，没有告诉它哪些错误路径会失败；RFT 只学成功轨迹，也浪费了失败轨迹里的负反馈；普通 DPO 只比较整条轨迹，仍然不知道中间哪一步把任务带偏。Agent Q 的妙处是把网页交互过程展开成搜索树：同一页面下的多个动作可以直接比较，搜索树告诉它哪些分支最后成功，critic 告诉它中间步骤谁更靠谱，再用 DPO 把这些偏好写进模型参数里。这样它不只是“会模仿成功案例”，还学到“为什么某些看似合理的点击其实不该选”。

## 2. 实验设置与结果 (Experiments)
- **基准测试 (Benchmarks)**：Agent Q 评估两个环境。第一个是 WebShop，一个模拟电商网站 benchmark，任务是根据自然语言需求找到精确商品，指标是 exact product match success rate。论文使用 12,087 个预定义任务，其中 11,000 个作为训练任务，1,087 个 held-out tasks 用于 zero-shot evaluation；WebShop 平均轨迹长度约 6.8 步。第二个是 OpenTable live booking 场景，任务是在真实 OpenTable 网站上为用户预订餐厅，要求找到餐厅、选择日期时间、人数、座位偏好并填写用户信息；OpenTable 平均需要 13.9 步，是 WebShop 的两倍多。OpenTable 的 reward 由 GPT-4V evaluator 根据最终截图和压缩轨迹历史判定。

- **对比基线 (Baselines)**：WebShop 中基础模型是 xLAM-v0.1-r。对比包括 base xLAM、RFT、trajectory-level outcome DPO、base + MCTS、Agent Q policy-only、Agent Q + MCTS，以及平均人类水平。OpenTable 中基础模型从 xLAM 切到 LLaMA-3-70B-Instruct，因为 xLAM 在真实网站任务上为 0.0%。对比包括 LLaMA-3-70B zero-shot、GPT-4o zero-shot、RFT on 600 successful trajectories、outcome-only DPO、MCTS outcome-only Q without process critique、Agent Q、RFT + MCTS、Agent Q + MCTS。

- **核心量化指标**：WebShop 的主指标是成功率。论文 Figure 3 caption 报告 base xLAM 为 28.6%、RFT 为 31.3%、DPO 为 37.5%、平均人类为 50.0%、Agent Q + MCTS 为 50.5%；正文又描述 outcome DPO 可到 40.6%，并说明 base + MCTS 为 48.4%。由于论文内部对 DPO 数字存在 caption/body 口径差异，下表保留两种原文口径。

| WebShop 方法 | Success Rate | 说明 |
|---|---:|---|
| xLAM-v0.1-r base | 28.6 | 基础 web agent policy |
| RFT | 31.3 | 成功轨迹过滤后做 SFT/STaR-like RFT |
| Outcome DPO | 37.5 / 40.6 | Figure caption 为 37.5；正文称比 RFT 高 9.3 到 40.6 |
| Base + MCTS | 48.4 | 只加推理时搜索，接近平均人类 |
| Average Human | 50.0 | WebShop 平均人类成功率 |
| Agent Q + MCTS | 50.5 | 训练后 policy 再配 MCTS，略超平均人类 |

OpenTable 的结果更能体现 Agent Q 的价值，因为它更长、更接近真实网站且 reward 更难获得。论文 Figure 6 和正文给出如下数据：

| OpenTable 方法 | Success Rate | 说明 |
|---|---:|---|
| xLAM-v0.1-r | 0.0 | 无法适应真实开放网页 |
| LLaMA-3-70B-Instruct zero-shot | 18.6 | 初始 policy |
| GPT-4o zero-shot | 62.6 | 闭源强基线 |
| RFT on 600 successful trajectories | 67.2 | 单轮成功轨迹监督微调 |
| Outcome-only DPO | 71.8 | 只用整条成功/失败轨迹偏好 |
| MCTS Q without process critique | 75.2 | 有树搜索分支 credit assignment，但没有 AI process feedback |
| Agent Q policy-only | 81.7 | MCTS empirical value + AI critique 构造 step preference 后 DPO |
| RFT + MCTS | 84.3 | RFT policy 加推理时搜索 |
| Agent Q + MCTS | 95.4 | 最强结果，训练后 policy 再加在线搜索 |

- **消融实验 (Ablation Studies)**：第一，WebShop 中 RFT 只从 28.6 提到 31.3，说明只复制成功轨迹无法解决探索；DPO 明显强于 RFT，因为失败轨迹也能作为 rejected signal。第二，base + MCTS 从 28.6 到 48.4，说明推理时搜索本身很强；但仅搜索不训练会产生大量交互成本，且不能把搜索经验固化到 policy。第三，OpenTable 中 outcome-only DPO 为 71.8，而加入 MCTS branching 但不用 process critique 为 75.2，说明树搜索能提供更细粒度 credit assignment；完整 Agent Q 为 81.7，比无 process critique 高 6.5 个点，说明 self-critique ranking 对长任务有明显帮助。第四，Agent Q + MCTS 达到 95.4，显示“训练后的 policy 能力”和“推理时搜索计算”可以叠加；读实验时必须区分 policy-only 的 81.7 和 policy+search 的 95.4。

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/agentq_arch.png)

![MCTS 搜索树流程图](./images/agentq_mcts_tree.png)

![Q_feedback 与偏好对构造图](./images/agentq_q_feedback.png)

![WebShop 结果图](./images/agentq_webshop_results.png)

![OpenTable evaluator 与结果图](./images/agentq_opentable_results.png)

## 4. 局限性与未来方向 (Limitations)
- Agent Q 的 MCTS 需要在真实或模拟网页中反复试错。对 WebShop 这类可重置环境较安全，但在 OpenTable、支付、邮件、提交表单等真实网站上，大量错误交互可能难以回滚，也可能产生安全风险。
- 论文使用的 critic/self-critique 主要是 prompt 出来的 ranking model，并没有联合训练 critic。作者自己也指出 frozen critic 可能限制 search quality，未来可以训练专门的 process reward model 或 value model。
- OpenTable reward 依赖 GPT-4V evaluator。虽然这种 evaluator 能处理真实网页最终状态，但它是闭源、成本高、可能误判的组件；如果 evaluator 错了，偏好对和 DPO 都会被污染。
- 结果集中在 WebShop 和 OpenTable，不能直接推出对通用网页、企业 SaaS、桌面 GUI 或 OS-level computer-use 任务都有效。尤其是 OpenTable 是一个窄域预订场景，动作空间和 reward 条件可以被相对明确地定义。
- Agent Q 的训练更像 off-policy preference learning，而不是在线 PPO/GRPO。它可以利用成功和失败轨迹，但策略更新没有直接按当前 policy 的 on-policy advantage 做，因此如果环境分布变化很快，旧 replay buffer 中的偏好可能过时。
- 未来方向包括：学习可泛化的 web process reward model；减少 MCTS 对真实环境的危险交互；引入 sandbox/rollback；把搜索数据生成和 policy update 做成更稳定的持续学习系统；在 BrowserGym/WebArena/真实 SaaS 上做更广泛评估。
