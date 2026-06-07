# DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning (2024 - Hao Bai, Yifei Zhou, Mert Cemri, Jiayi Pan, Alane Suhr, Sergey Levine, Aviral Kumar)

## 1. 核心方法 (Methods)
### 🗺️ 架构全局观（系统是怎么拼起来的）
- **一句话本质**：DigiRL 核心解决的是“手机操作里，失败的尝试也是训练材料”的问题。它不是只模仿成功轨迹，而是先让 agent 自己在 Android emulator 里跑出成功和失败轨迹，再训练两个 Value 网络当裁判：一个判断“这个任务值不值得练”，另一个判断“这一步动作有没有把任务往成功方向推进”，最后只让 Actor 模仿这些被筛出来的高价值动作。

- **全局流水线**：

```text
任务指令 c + Android emulator 初始 home screen
        ↓
Actor πθ 根据当前截图/任务输出动作文本，例如 DUAL_POINT / TYPE / BACK
        ↓
Android emulator 真实执行动作，得到下一张截图 s_{t+1}
        ↓
一条轨迹 τ = (s0,a0,r0,...,sH,aH,rH) 跑完或提前成功
        ↓
Gemini 1.5 Pro evaluator 看任务和当前/最终截图，给 sparse reward
        ↓
轨迹进入 replay buffer：保存 observation、action token、image_features、reward、MC return
        ↓
训练两个 Value 网络：
  1. Instruction-level Value：这个任务对当前策略来说成功概率多大？
  2. Step-level Value：从这个截图状态继续做，最终成功概率多大？
        ↓
用 Value 计算 advantage / mask：
  1. 任务层过滤：优先练“当前策略刚好有学习信号”的任务
  2. 步骤层过滤：只保留让成功概率上升的动作
        ↓
Actor 用带权/过滤的 MLE 行为克隆损失更新
```

这里的关键是：DigiRL 形式上是强化学习，因为它有环境交互、reward、trajectory、value、advantage 和 replay buffer；但它的 Actor 更新不像 PPO 那样直接做 on-policy policy gradient，而是把 RL 估计出来的 advantage 转成“哪些动作值得模仿”的权重或二值 mask，再做 Maximum Likelihood / Behavior Cloning。

### 🧠 强化学习三要素（State, Action, Reward 的技术死磕）
- **状态空间 State：它到底“看见”了什么？**

  论文把手机控制建模成有限时长 MDP：$M=\{S,A,T,\mu_0,R,H\}$。每个 episode 开始时，从任务集合里采样一条自然语言指令 $c$，emulator 初始化到 home screen，然后 agent 连续操作手机，最多走 $H$ 步；AitW General 的 $H=10$，Web Shopping 的 $H=20$。

  状态不是 DOM 树，也不是 accessibility tree，而是**像素截图**。论文明确说 states are represented using the last two screenshots，也就是状态表征里包含最近两帧：上一帧 $s_{t-1}$ 和当前帧 $s_t$。这样做的意义很实际：手机界面经常有加载、弹窗、输入框展开、页面跳转，如果只看当前截图，模型很难判断“我刚才点的动作到底有没有产生变化”；看两帧就能捕捉到“点完以后页面有没有动”。

  更细地说，官方实现不是把两张 RGB 图直接按通道拼成 6-channel 图片。它采用的是 **feature-level frame stacking**：

  1. Android emulator 配置里的原始屏幕是 `1080 × 2280`，竖屏，density 440。
  2. 每张截图先经过冻结视觉编码器提特征。论文口径是冻结 AutoUI 的 image encoder；公开代码里 `ImageFeatureExtractor` 使用 BLIP-2 vision path，把每张 PIL screenshot 编成一个 `1408` 维 `image_features` 向量。模型侧具体 resize 由 processor 内部处理，论文没有公开一个固定输入尺寸。
  3. Step-level Value 使用最近两帧特征拼接：`[feat(s_{t-1}); feat(s_t)]`，所以视觉部分是 `1408 × 2 = 2816` 维。第 0 步没有上一帧时，就用 `[feat(s_0); feat(s_0)]`。
  4. 文本任务/observation 用 RoBERTa 编成 `768` 维 `pooler_output`。
  5. 最后把 `RoBERTa text feature` 和 `2816-d image feature` concat，送进 MLP 预测 value。

  可以把这个状态理解成：“当前任务说明 + 当前屏幕 + 上一屏幕变化痕迹”。这比一句“看截图”重要得多，因为 DigiRL 的 step-level value 正是靠“前后截图变化”判断一个动作是不是有用。

- **动作空间 Action：它怎么真的控制手机？**

  Actor 是 AutoUI-style VLM policy。它不是调用高级 API，也不是点网页 DOM 元素，而是**自回归生成一串动作文本 token**。生成时，模型会先看到 prompt，例如最近动作历史和 `Goal: <task>`，再逐 token 输出 `Action Decision: ...` 后面的动作描述。训练 Actor 时，损失就是让模型提高这些动作 token 的 log probability。

  官方 AutoUI 动作文本格式大致如下：

```text
Action Decision: "action_type": "DUAL_POINT",
                 "touch_point": "[y, x]",
                 "lift_point": "[y, x]",
                 "typed_text": ""

Action Decision: "action_type": "TYPE",
                 "touch_point": "[-1.0, -1.0]",
                 "lift_point": "[-1.0, -1.0]",
                 "typed_text": "macbook air"

Action Decision: "action_type": "PRESS_BACK",
                 "touch_point": "[-1.0, -1.0]",
                 "lift_point": "[-1.0, -1.0]",
                 "typed_text": ""
```

  底层 action set 包括：

  - `DUAL_POINT`：两个归一化坐标点。`touch_point` 和 `lift_point` 几乎一样时就是点击；两点距离较远时就是滑动。注意文本里写的是 `[y, x]`，代码解析后会反转成 `(x, y)`，再乘以真实屏幕宽高执行。
  - `TYPE`：输入任意字符串。
  - `PRESS_HOME`、`PRESS_BACK`、`PRESS_ENTER`：Android 系统按键。
  - `STATUS_TASK_COMPLETE`、`TASK_IMPOSSIBLE`：模型认为任务完成或无法完成时的终止类动作。

  所以 DigiRL 的动作粒度是“人类手指级别”的 Android primitive actions：点坐标、滑屏、打字、返回、回主页、回车。它没有 SoM 那种强制编号按钮机制，也没有直接保证一定能点中正确控件；如果坐标预测错了，环境就真的执行错，后面的 Value/Reward 再从轨迹结果里纠正策略。

- **奖励 Reward：它怎么知道任务成功了？**

  奖励是稀疏的 binary reward。中间步骤默认没有 dense reward，只有当轨迹结束或某一步已经满足任务时，evaluator 才给成功/失败：

$$
r_t =
\begin{cases}
1, & \text{如果当前/最终截图已经完成任务}\\
0, & \text{否则}
\end{cases}
$$

  论文表述是：reward of 1 is given at the end if the agent successfully fulfills the task, otherwise 0；轨迹在成功或超过最大步数 $H$ 时终止。实现里 evaluator 可以在每一步后被调用，用来提前发现“已经完成了”，但语义仍然是 sparse：没完成的中间状态都是 0，成功那一步才是 1。

  VLM evaluator 用 Gemini 1.5 Pro。它的 prompt 逻辑不是让 Gemini 自由发挥长篇解释，而是给 few-shot 示例，然后要求它按固定格式回答：

```text
Q: What should I expect to see on the screenshot if I've <task>?
A: I should expect to see <先描述成功时截图应该长什么样，再描述当前截图实际是什么>.
Status: success or failure
```

  也就是说，Gemini 不是给每一步打“进度分”，而是做二分类：当前截图是否足以证明任务已经完成。官方代码还做了两个防误判处理：如果当前截图和上一张几乎没变化，直接跳过评估；如果截图在最近历史里重复出现，也跳过评估。这是为了避免页面没加载、卡住或重复状态被误判成成功。

### ⚙️ 核心算法：AWR 与双层价值网络（论文真正的 RL 机制）
- **先看 DigiRL 为什么不是普通 BC，也不是 PPO**

  普通 Behavior Cloning 的损失是：

$$
\mathcal{L}_{BC}(\theta)=-\mathbb{E}_{(s,a,c)\sim D}\left[\log \pi_\theta(a|s,c)\right]
$$

  它的问题是：只要数据里有这个动作，它就模仿，不管这个动作是好是坏。Filtered BC 稍微进一步：只模仿成功轨迹。但这又会浪费失败轨迹里“前几步其实很好、最后一步搞砸了”的片段。

  DigiRL 的出发点是 Advantage Weighted Regression (AWR)：

$$
\max_\pi\ \mathbb{E}_{(s,a,c)\sim \nu}
\left[
\log \pi(a|s,c)\cdot \exp\left(\frac{A(s,a,c)}{\beta}\right)
\right]
$$

  这句话翻译成人话就是：同样是模仿历史动作，**优势越大的动作，模仿权重越大**。如果 $A>0$，说明这个动作比当前策略平均水平更好；如果 $A<0$，说明它让局面变差，不该学。

  不过论文最后没有强依赖连续的 $\exp(A/\beta)$ 权重，因为 GUI 环境随机性很强，$\beta$ 很难调，value 也有噪声。DigiRL 实际主算法采用 hard filtering：

$$
\mathcal{L}_{actor}(\theta)
=-\mathbb{E}_{(s,a,c)\sim filter(\nu)}
\left[\log \pi_\theta(a|s,c)\right]
$$

  也就是：先用 advantage 把“值得学”的样本筛出来，然后只在这些样本上做 MLE。你可以把它看成 AWR 的二值版本：原始 AWR 的权重是 $w=\exp(A/\beta)$；DigiRL 的工程版本近似成 $w=\mathbf{1}[A>\text{threshold}]$。

- **Instruction-level Value $V_\phi(c)$：任务挑选器**

  它输入任务指令 $c$，输出一个概率：**当前策略做这个任务，大概有多大概率最终成功**。例如：

  - “打开设置”如果已经 95% 成功，继续练它没有太大意义。
  - “去 costco 搜一个很冷门商品并点第一个结果”如果 0% 成功，也可能暂时太难，采样再多也全是负反馈。
  - 成功率在中间的任务最值钱，因为 agent 正在卡关，轨迹里最可能出现可学习信号。

  训练标签来自整条轨迹最后的二值结果：

$$
y(\tau)=r_T\in\{0,1\}
$$

  论文没有把它训练成 MSE 回归，而是用 binary cross-entropy / cross-entropy 来学成功概率：

$$
\mathcal{L}_{traj}(\phi)=
-\mathbb{E}_{\tau\sim\nu}
\left[
y(\tau)\log V_\phi(c)
+(1-y(\tau))\log(1-V_\phi(c))
\right]
$$

  训练好以后，用它算 instruction-level advantage：

$$
A_{instruct}(\tau,c)=r_T - V_\phi(c)
$$

  这个公式非常直观：如果一个任务本来预测成功率只有 0.1，但这次 rollout 成功了，$A_{instruct}=1-0.1=0.9$，说明这是一次“困难任务成功”的高价值经验，应该优先放进 Actor 训练。相反，如果一个任务本来预测成功率 0.95，这次也成功了，$A_{instruct}=0.05$，它虽然成功，但信息量不大。

- **Step-level Value $V_\psi(s,c)$：动作鉴宝师**

  它输入当前状态 $s_h$ 和任务 $c$，输出从这一步继续往后走最终成功的概率。论文公式有时写成 $V^{step}(s_h,a_h,c)$，但核心含义是对“当前步骤附近的状态/动作上下文”估计成功概率；官方 critic 的 forward 接口里确实接收 `action` 参数，但网络主体主要使用 RoBERTa observation feature 和两帧 image feature 拼接来预测 success/failure。

  网络结构上，Step-level Value 是 double critic：

```text
text observation / task  --RoBERTa--> 768-d text feature
previous screenshot     --frozen vision encoder--> 1408-d image feature
current screenshot      --frozen vision encoder--> 1408-d image feature
concat = 768 + 1408 + 1408
        ↓
MLP critic 1: Linear -> ReLU -> Linear -> ReLU -> Linear(2 logits)
MLP critic 2: Linear -> ReLU -> Linear -> ReLU -> Linear(2 logits)
        ↓
softmax(success logit) = V_step
```

  为什么要两个 critic？这是深度 RL 里常见的保守技巧：两个网络都预测成功概率，Actor 计算 advantage 时取较小值 `min(V1,V2)`，避免某一个 value head 过度乐观，把坏动作误判成好动作。

  它的训练目标同样是最终成功/失败的 cross-entropy，不是每步人工打标签：

$$
\mathcal{L}_{step}(\psi)=
-\mathbb{E}_{(s_h,a_h,c)\sim\nu}
\left[
r_T\log V_\psi(s_h,c)
+(1-r_T)\log(1-V_\psi(s_h,c))
\right]
$$

  这里容易误会：每个中间状态都没有自己的人工 reward，但它继承整条轨迹的最终标签。成功轨迹里的每一步都被标成“从这里出发后来成功了”，失败轨迹里的每一步都被标成“从这里出发后来失败了”。这看起来很粗，但 replay buffer 多了以后，Value 网络会学到统计规律：哪些截图状态通常通向成功，哪些通常通向失败。

- **Step Advantage：怎么判断一个动作是不是推进了任务？**

  最朴素的做法是用：

$$
A_h \approx V_\psi(s_{h+1},c)+r_h-V_\psi(s_h,c)
$$

  这就是一阶 TD 思想：如果做完动作后，下一状态的成功概率比当前状态高，说明这个动作有帮助。比如当前页面成功概率 0.10，点进正确搜索结果后下一页成功概率 0.80，那么 $A_h\approx0.80-0.10=0.70$，这个动作就值得模仿。

  但 GUI 环境很随机：网页加载慢、广告弹窗、搜索结果排序变动都会让单步差分很吵。因此 DigiRL 把两类信号混起来：

$$
A_{step}(s_h,a_h,c)
=
\lambda^{H-h}r_T
+
(1-\lambda^{H-h})
\left[
V_\psi(s_{h+1},c)+r_h-V_\psi(s_h,c)
\right]
$$

  这就是论文说的 doubly-robust / simplified GAE 思路：

  - $r_T$ 是 Monte Carlo 终局信号：优点是真实，缺点是方差大。
  - $V_\psi(s_{h+1},c)-V_\psi(s_h,c)$ 是 TD-like 局部进展信号：优点是能识别失败轨迹里的好前缀，缺点是依赖 value 估计，可能有偏。
  - $\lambda$ 用来平衡两者。离终点远时，终局 reward 对某一步动作的责任更模糊；靠近终点时，终局 reward 更可信。

  论文的 hard filtering 阈值是：

$$
A_{step}(s_h,a_h,c) > \frac{1}{H}
$$

  满足这个条件的动作才被认为“推进了任务”。官方训练代码里还有一个更工程化的 mask 版本：

```text
v  = min(V1(s_h),     V2(s_h))
nv = min(V1(s_{h+1}), V2(s_{h+1}))
advantage = nv - v - 0.05 + reward + mc_return
advantage = clamp(advantage, 0, 1)
advantage_mask = 1[advantage > 0]
actor_loss = -mean(log πθ(a_h | s_h,c) * advantage_mask)
```

  这段代码的意思是：如果下一步 value 上升，或者当前步拿到成功 reward，或者整条轨迹最终成功，就把这个动作当成正样本；否则不给 Actor 学。`-0.05` 是一个小惩罚，避免“几乎没进展”的动作也被算成正优势。

- **为什么它能从失败里学习？**

  假设一个购物任务最终失败了：前 3 步先打开正确网站、搜索商品、点进正确结果，第 4 步误点广告导致失败。Filtered BC 会把整条失败轨迹扔掉；普通 BC 可能连误点广告也学进去。DigiRL 的 step value 会看到：

```text
s0: home screen                 V_step = 0.10
s1: 正确打开 walmart.com        V_step = 0.35
s2: 正确输入商品并出现结果       V_step = 0.65
s3: 点进正确商品页              V_step = 0.80
s4: 误点广告页                  V_step = 0.05
```

  那么前 3 步的 $V(s_{h+1})-V(s_h)$ 是正的，会被保留下来；第 4 步让 value 从 0.80 掉到 0.05，会被过滤掉。这就是“失败的尝试也是宝贝”的数学含义：失败轨迹不是整条有用，而是里面有些动作有用，有些动作有害，DigiRL 用 value 差分把它们拆开了。

### 💻 训练流程硬核伪代码（从 offline 到 offline-to-online）
```python
# 初始化
actor = AutoUI_Base_policy()          # 已经 SFT 过，会输出 Android action text
actor.freeze_image_encoder()          # 论文实验里冻结 image encoder
V_instr = InstructionValue_RoBERTa()  # 输入任务/observation，输出 success probability
V_step  = DoubleStepValue()           # 输入 RoBERTa text feature + 两帧 image feature，输出 success probability
replay_buffer = ReplayBuffer(capacity=5000)

# ----------------------------
# Phase 1: Offline pre-training
# ----------------------------
# 用旧策略/初始 AutoUI 策略在 AitW 任务上收集离线轨迹。
# 每条轨迹包含成功和失败，不只保留成功。
offline_trajectories = collect_or_load_trajectories(policy=actor, env=android_emulator)

for tau in offline_trajectories:
    # tau = [(s0,a0,r0), ..., (sT,aT,rT)]
    # rT 来自 Gemini evaluator；中间 r 通常为 0
    tau.mc_return = tau.final_reward  # 因为没有中间 dense reward，所以 MC return 基本就是 rT
    replay_buffer.add(tau)

# 1. 训练 instruction-level value：学 P(success | instruction)
for update in range(num_instr_value_updates):
    batch_tau = replay_buffer.sample_trajectories()
    y = batch_tau.final_reward  # 0/1
    pred = V_instr(batch_tau.instruction)
    loss_instr = BCE(pred, y)
    optimize(V_instr, loss_instr)

# 2. 训练 step-level value：学 P(success | state, instruction)
for update in range(num_step_value_updates):
    batch_steps = replay_buffer.sample_steps()
    y = batch_steps.trajectory_final_reward  # 这一步所在轨迹最终是否成功
    pred1, pred2 = V_step(batch_steps.prev_image,
                          batch_steps.curr_image,
                          batch_steps.instruction)
    loss_step = CE(pred1, y) + CE(pred2, y)
    optimize(V_step, loss_step)

# 3. 更新 actor：offline 阶段主要用已有数据冷启动。
# 论文说明 offline 阶段为了充分使用有限数据，会弱化/跳过 instruction-level 过滤；
# actor 本质上仍是用 value/成功信号筛过的数据做 MLE。
for update in range(num_actor_updates):
    batch_steps = replay_buffer.sample_steps()

    with no_grad():
        v  = min(V_step(batch_steps.s).success_prob)
        nv = min(V_step(batch_steps.s_next).success_prob)
        A_step = compute_doubly_robust_advantage(v, nv,
                                                 reward=batch_steps.r,
                                                 final_reward=batch_steps.final_reward,
                                                 lambda_=lambda_)
        mask = (A_step > 1 / H)

    logp = actor.log_prob(batch_steps.action_tokens,
                          condition=(batch_steps.s, batch_steps.instruction))
    loss_actor = -mean(logp * mask)
    optimize(actor, loss_actor)

# ---------------------------------
# Phase 2: Offline-to-online training
# ---------------------------------
for iteration in range(num_online_iterations):
    # A. 当前 actor 真实控制 Android emulator，收集新轨迹
    new_trajectories = []
    for env in parallel_android_emulators:
        c = sample_instruction()
        s = env.reset_to_home()
        tau = []

        for h in range(H):
            # actor 自回归生成动作 token
            action_text = actor.generate_action(observation=s, instruction=c)
            a = parse_autoui_action(action_text)

            # emulator 执行动作，得到下一张截图
            s_next = env.step(a)

            # evaluator 判断是否完成；没完成就是 0，完成就是 1 并终止
            r = gemini_evaluator(task=c, screenshot=s_next)
            tau.append((s, a, r, s_next))

            if r == 1:
                break
            s = s_next

        tau.final_reward = tau[-1].reward
        new_trajectories.append(tau)

    replay_buffer.add(new_trajectories)

    # B. 用 replay buffer 重新训练/刷新两个 value
    for update in range(num_instr_value_updates):
        batch_tau = replay_buffer.sample_trajectories()
        y = batch_tau.final_reward
        loss_instr = BCE(V_instr(batch_tau.instruction), y)
        optimize(V_instr, loss_instr)

    for update in range(num_step_value_updates):
        batch_steps = replay_buffer.sample_steps()
        y = batch_steps.trajectory_final_reward
        p1, p2 = V_step(batch_steps.s)
        loss_step = CE(p1, y) + CE(p2, y)
        optimize(V_step, loss_step)

    # C. 任务层 curriculum：优先选“比预期更成功”的轨迹
    candidate_tau = replay_buffer.sample_trajectories()
    A_instr = candidate_tau.final_reward - V_instr(candidate_tau.instruction)
    selected_tau = top_p(candidate_tau, score=A_instr)

    # D. 步骤层 advantage：在 selected_tau 中继续挑动作
    selected_steps = flatten(selected_tau)
    with no_grad():
        v  = min(V_step(selected_steps.s).success_prob)
        nv = min(V_step(selected_steps.s_next).success_prob)

        # 论文版：soft AWR 权重
        A = lambda_ ** (H - selected_steps.h) * selected_steps.final_reward \
            + (1 - lambda_ ** (H - selected_steps.h)) \
              * (nv + selected_steps.reward - v)
        w_soft = exp(A / beta)

        # DigiRL 主实现：hard filtering，避免调 beta
        w_hard = (A > 1 / H).float()

    # E. Actor 更新：不是“盲目模仿”，而是 advantage-weighted / filtered MLE
    logp = actor.log_prob(selected_steps.action_tokens,
                          condition=(selected_steps.s, selected_steps.instruction))

    # 概念 AWR：
    # loss_actor = -mean(logp * w_soft)
    #
    # DigiRL 实际采用的硬过滤版本：
    loss_actor = -mean(logp * w_hard)
    optimize(actor, loss_actor)
```

## 2. 实验设置与结果 (Experiments)
- **基准测试 (Benchmarks)**：DigiRL 只在 Android 设备控制任务上评估，核心 benchmark 是 Android-in-the-Wild (AitW) 的两个多步子集。General 子集是开放信息查询和基础 app 使用任务，例如“最近的 Verizon 店怎么走”“纽约有什么好餐厅”“Burger King 菜单有什么”；训练集 545 个任务，测试使用前 96 个任务，最大步数 10。Web Shopping 子集是购物网站任务，作者过滤掉 captcha 和需要账号/购物车的操作，保留 costco.com、bestbuy.com、target.com、walmart.com、newegg.com 等网站上的“进入网站、搜索、选择结果”任务；训练集 438 个任务，测试 96 个任务，最大步数 20。Install 和 GoogleApps 子集因账号/安全问题未使用，Single 子集因一步即可完成，不能体现多步决策难度，也未使用。

- **对比基线 (Baselines)**：Prompting 类包括 Set-of-Marks + GPT-4V、Set-of-Marks + Gemini 1.5 Pro，以及 AppAgent + GPT-4V/Gemini 1.5 Pro。监督训练类包括 AutoUI-Base 和 CogAgent。学习类包括 Filtered BC，以及 DigiRL 的 offline 和 offline-to-online 两种训练设置。Filtered BC 的逻辑是只模仿成功轨迹，DigiRL 则用 value function 判断轨迹/步骤是否有学习价值。DigiRL 和 Filtered BC 都从 AutoUI-Base checkpoint 出发，冻结 image encoder；value function 使用同一 frozen image encoder 的视觉特征，加 RoBERTa instruction feature，再接两层 MLP。

- **核心量化指标**：主指标是 task success rate，由 autonomous evaluator 在测试任务上判定。表中数值为论文 Table 1 的成功率，学习类为多次运行均值和标准差。

| 方法类别 | 方法 | AitW General Train | AitW General Test | AitW Web Shopping Train | AitW Web Shopping Test |
|---|---|---:|---:|---:|---:|
| Prompting | Set-of-Marks + GPT-4V | 5.2 | 13.5 | 3.1 | 8.3 |
| Prompting | Set-of-Marks + Gemini 1.5 Pro | 32.3 | 16.7 | 6.3 | 11.5 |
| Prompting | AppAgent + GPT-4V | 13.5 | 17.7 | 12.5 | 8.3 |
| Prompting | AppAgent + Gemini 1.5 Pro | 14.6 | 16.7 | 5.2 | 8.3 |
| Supervised Training | CogAgent | 25.0 | 25.0 | 31.3 | 38.5 |
| Supervised Training | AutoUI | 12.5 | 14.6 | 14.6 | 17.7 |
| Learning Offline | Filtered BC | 51.7 +/- 5.4 | 50.7 +/- 1.8 | 44.7 +/- 1.6 | 45.8 +/- 0.9 |
| Learning Offline | DigiRL | 46.9 +/- 5.6 | 62.8 +/- 1.0 | 39.3 +/- 6.0 | 45.8 +/- 6.6 |
| Learning Offline-to-Online | Filtered BC | 53.5 +/- 0.8 | 61.5 +/- 1.1 | 53.6 +/- 4.7 | 57.8 +/- 2.6 |
| Learning Offline-to-Online | DigiRL | 63.5 +/- 0.0 | 71.9 +/- 1.1 | 68.2 +/- 6.8 | 67.2 +/- 1.5 |

从结果看，DigiRL 的 offline-to-online 版本在 General Test 上达到 71.9%，在 Web Shopping Test 上达到 67.2%，显著高于纯 SFT 的 AutoUI，也高于 Filtered BC。论文摘要中特别强调 Web Shopping 上从 AutoUI/SFT 的 17.7% 提升到 67.2%，绝对提升 49.5%。和 CogAgent 的 38.5% 相比，DigiRL 在 Web Shopping Test 也高出约 28.7 个百分点。

- **消融实验 (Ablation Studies)**：第一，value function 用 binary cross-entropy 比 regression 更好，作者报告约 12% 提升，说明稀疏成功/失败标签下把 value 当成功概率分类问题更稳定。第二，step-level advantage 提高样本效率，约 12% 效率提升，说明只按整条成功轨迹做 BC 会丢掉“失败轨迹中前面几步其实正确”的信号。第三，automatic curriculum 提升学习速度约 25%，说明任务难度调度对 online GUI RL 很关键；太简单任务没有增益，太难任务几乎不给正反馈。第四，DigiRL 优于 vanilla AWR，原因是 vanilla AWR 没有 doubly-robust advantage estimator 和 curriculum。第五，环境系统也做了消融：分布式 emulator 在 64 CPUs 下达到约 1.74 trajectories/min，而 naive 单实例并行约 0.74 trajectories/min；这说明论文的贡献不只是算法，rollout 吞吐也是 online RL 可行性的关键。作者还分析了失败模式，DigiRL 对“误操作后无法恢复”的失败类型改善最明显，说明在线轨迹确实让模型学到了一些 recovery 行为。

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/digirl_arch.png)

![Android 环境与动作空间](./images/digirl_env_action_space.png)

![训练算法流程图](./images/digirl_training_flow.png)

![主结果表格截图](./images/digirl_main_results.png)

![消融实验曲线](./images/digirl_ablation.png)

## 4. 局限性与未来方向 (Limitations)
- DigiRL 的结论主要来自 AitW General 和 Web Shopping 两个 Android 子集，并没有证明同一训练方案可以直接迁移到 Windows、Linux、VSCode、Office 或通用 OS agent 任务。论文自己也说明由于计算限制，虽然环境和 evaluator 可以扩展，实际训练只覆盖 AitW 的部分任务。
- 奖励依赖 Gemini 1.5 Pro evaluator。虽然作者做了人类相关性验证并报告平均错误率约 2.8%，但 reward model 一旦误判，actor 会学习错误偏好；对于更复杂任务，VLM evaluator 的可靠性会成为瓶颈。
- Online RL 成本高。DigiRL 需要并行 emulator、replay buffer、持续 evaluator 调用和真实网页/app 状态重置，这比离线 SFT 或离线 Q-learning 难复现，也更容易受网站更新、captcha、账号限制和网络状态影响。
- 动作空间仍是 Android primitive actions。它不包含文件系统、terminal、IDE 编辑、系统级 API 调用等复杂 computer-use 动作，因此不能直接说明它能处理“在 VSCode 中修改某行代码”这类长程桌面任务。
- 未来方向是把 DigiRL 的 offline-to-online 框架扩展到更通用的 GUI sandbox，替换或校准 VLM evaluator，加入更强的安全约束和可回滚机制，并研究比 hard-filtered AWR 更稳定、更可扩展的多轮 actor-critic 或 PPO/GRPO 训练。
