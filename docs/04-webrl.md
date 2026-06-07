# WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning (2025 - Zhilin Wang et al.)

## 1. 核心方法 (Methods)
- 💡**用一句话大白话概括它的核心创新点**：WebRL 是把网页任务做成“自己出题、自己上网做题、自己验收、再继续训练”的闭环，让开源 LLM 通过在线课程学习从不会用网页逐步变成能完成 WebArena 任务的 agent。

- **它到底解决什么问题**：WebArena 这种网页环境很难直接做 RL。第一，任务少，WebArena-Lite 只有约 1k 条带 oracle trajectory 的训练样本，不够把一个 8B/9B 模型训成强 web agent。第二，reward 很稀疏，网页任务通常只有最后成功/失败，没有每一步的进度分。第三，在线训练会导致策略漂移，模型刚学会当前阶段任务，就可能忘掉之前阶段的网页技能。WebRL 的做法是三件事合在一起：用失败任务生成新任务做 curriculum，用 ORM 判断终局成功，用 KL-constrained update 和 replay buffer 稳住策略。

- **系统是怎么运作的（工作流）**：

```text
1. 先用 WebArena-Lite 的人工轨迹做 SFT，得到一个会基本网页动作的初始 policy。
2. 让 SFT policy 在训练任务上 rollout，得到成功轨迹和失败轨迹。
3. 把成功轨迹放进 replay buffer，把失败任务放进 failure set。
4. 每个 curriculum phase：
   a. 用 GPT-4o 根据失败任务生成一批新 instruction。
   b. 用 critic/value score 过滤难度：只保留 score 在 0.05 到 0.75 的任务。
   c. 再用 GPT-4o prompt 去掉 WebArena 中不可执行的任务。
   d. 当前 policy 在新任务上真实执行网页操作，产生 rollout。
   e. ORM 根据 instruction、历史动作、最终 HTML 判断 success/failure。
   f. 从 replay buffer 取“当前模型既不太熟也不太陌生”的成功经验。
   g. 用 KL-constrained RL loss 更新 actor，用 CE loss 更新 value。
   h. 新成功轨迹继续写回 replay buffer，新失败任务继续进入 failure set。
```

- **AI 怎么看（State）**：WebRL 不是直接看截图坐标，也不是像 GUI-R1 那样只做视觉 grounding。它的状态 $s_t$ 主要是三块文字化信息：用户 instruction $I$、action history、当前网页 HTML。作者会简化 HTML 结构，并给可点击元素分配 element ID，让模型输出动作时能引用具体元素。这个设计的本质是把浏览器页面变成一个可读、可操作的文本环境：模型不需要猜坐标，而是读 HTML 里的按钮、输入框、链接和文本，然后选择 element ID。

- **AI 怎么动（Action）**：动作是 WebArena-style browser action，不是像 Android 那样输出 `(x,y)` 坐标。论文附录给出的 `do(action, argument, element)` 形式包括：

```python
do(action="Click", element="7")
do(action="Type", argument="macbook air", element="12")
do(action="Search", argument="winter coat", element="4")
do(action="Scroll Down")
do(action="Select Dropdown Option", argument="Month", element="20")
exit(message="final answer")
```

  具体 action set 包括 `Click`、`Right Click`、`Type`、`Search`、`Hover`、`Scroll Up/Down`、`Press Enter`、`Switch Tab`、`Select Dropdown Option`、`Wait`、`Goto`、`Go back/forward`、`Exit` 等。模型输出里还会附带 `# Element:` 和 `# Note:` 注释，用来说明正在操作哪个网页元素、当前网页中哪段信息支撑这个动作。

- **Reward 是怎么来的**：WebRL 的环境本身不能为新生成任务提供人工 reward function，所以作者训练了一个 outcome-supervised reward model，记作 $M_{ORM}$。它只判断终局 outcome，不给每一步打分。输入是：任务 instruction、完整历史动作、最终页面 HTML。输出只比较两个 token 的概率：`YES` 和 `NO`。如果 `P(YES)>P(NO)`，reward 记为 1，否则为 0。

$$
r(\tau, I)=
\begin{cases}
1, & M_{ORM}(I,\text{history},HTML_T)=YES\\
0, & M_{ORM}(I,\text{history},HTML_T)=NO
\end{cases}
$$

  为什么只看最终 HTML？因为完整网页 HTML 和整条轨迹太长，LLM context 放不下。作者借鉴 DigiRL 的 evaluator 思路，只把最后状态和历史动作压进 ORM，让它回答“这条轨迹最终有没有完成 instruction”。

- **训练算法不是普通 PPO，而是 KL-constrained off-policy update**：WebRL 的推导目标是让策略提高正 advantage 动作概率，同时不要偏离参考策略太远。论文给出的关键等式是：

$$
\beta \log \frac{\pi^*(a_t|s_t,I)}{\pi_{ref}(a_t|s_t,I)}
= r(s_t,a_t,I)+V^*(s_{t+1},I)-V^*(s_t,I)
= A^*(s_t,a_t,I)
$$

  直白解释：如果某个动作的 advantage 为正，它应该比参考策略更常被选；如果 advantage 为负，它应该被压低。但这个调整被 $\beta$ 控制，不能让 policy 一下子漂移太远。对应的 policy loss 是：

$$
L(\pi_\theta)=
\mathbb{E}_{\nu}
\left[
\left(
\beta \log \frac{\pi_\theta(a|s,I)}{\pi_{ref}(a|s,I)}
-A(s,a,I)
\right)^2
\right]
$$

  这和一句“做 RL”不一样：它不是简单把成功轨迹拿来 SFT，而是显式用 advantage 指挥概率上升或下降，并用 reference policy 做 KL 约束。

- **Value / Advantage 怎么训练**：因为中间步骤没有 reward，WebRL 用最终二值 outcome 训练 value network，采用 cross-entropy 而不是 MSE：

$$
L(V)=
-\mathbb{E}_{\nu}
\left[
r_T\log V(s,I)+(1-r_T)\log(1-V(s,I))
\right]
$$

  Advantage 结合“下一步进展”和“最终成败”两类信号：

$$
A(s_t,a_t,I)=
\lambda\left(r_t+V(s_{t+1},I)-V(s_t,I)\right)
+(1-\lambda)\left(r_T-V(s_t,I)\right)
$$

  论文设置 $\lambda=0.5$。大白话理解：如果点完按钮以后网页更接近成功，$V(s_{t+1})-V(s_t)$ 会变正；如果整条轨迹最后成功，$r_T$ 又会给整条路径一个终局正信号。两个信号结合，可以缓解“最后一步失败导致前面正确动作也被错杀”的问题。

- **Replay buffer 怎么选数据**：WebRL 只把成功轨迹放入 replay buffer。每个新 phase 训练时，用上一阶段 actor 计算 buffer 中动作的 perplexity，只取 perplexity 在 `[1/0.95, 1/0.5]` 的样本。太低表示模型已经很熟，继续学没价值；太高表示当前模型太不会，硬学容易造成 policy drift。这个 actor confidence filtering 本质上就是“只复习刚好卡住的旧经验”。

- 🎯**它解决了前人什么头疼的问题**：SFT 只会复读人工轨迹，Filtered BC 只学成功轨迹，DigiRL/AWR 直接搬到 WebArena 又容易因为任务少和网页稀疏 reward 不稳定。WebRL 的妙处是把失败转成课程生成信号：失败任务不是丢掉，而是用来生成下一阶段难度合适的新任务；旧成功经验不是无限重复，而是用 perplexity 选出当前模型最需要复习的部分；policy update 也不是裸 REINFORCE，而是带 KL 约束，避免每个 curriculum phase 把前一阶段技能洗掉。

## 2. 实验设置与结果 (Experiments)
- **基准测试 (Benchmarks)**：主环境是 WebArena / WebArena-Lite。WebArena 包含 Reddit、GitLab、CMS、OpenStreetMap、OneStopShop 等真实感网页站点，任务需要多轮浏览、搜索、填表、查信息或操作网页状态。WebArena-Lite 是人工验证过的子集，用于更可靠评估。训练起点使用 WebArena-Lite 的 1,186 条训练样本，ORM 训练时通过 instruction rewriting 和 baseline rollouts 扩展到 12,200 条带成功/失败标签的样本。

- **对比基线 (Baselines)**：闭源/框架类包括 GPT-4-Turbo、GPT-4o、AWM + GPT-4-0613、WebPilot + GPT-4o。开源类包括 AutoWebGLM、GLM-4-Chat、GLM-4 + SFT、Filtered BC、AWR、DigiRL、WebRL；另有 Llama3.1-8B/70B 同样比较 SFT、Filtered BC、AWR、DigiRL、WebRL。

- **核心量化指标**：主指标是 WebArena-Lite task success rate。论文 Table 1 的关键结果如下：

| 方法 | 参数量 | Reddit | GitLab | CMS | Map | OSS | Avg. SR |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-4-Turbo | N/A | 10.5 | 16.7 | 14.3 | 36.7 | 13.3 | 17.6 |
| GPT-4o | N/A | 10.5 | 10.0 | 20.0 | 20.0 | 11.1 | 13.9 |
| AutoWebGLM | 6B | 9.4 | 15.0 | 28.6 | 24.8 | 17.1 | 18.2 |
| GLM-4-Chat | 9B | 5.3 | 10.0 | 6.7 | 3.3 | 6.7 | 6.1 |
| GLM-4 + SFT | 9B | 47.4 | 13.3 | 31.4 | 23.3 | 13.3 | 22.4 |
| GLM-4 + Filtered BC | 9B | 52.6 | 10.0 | 31.4 | 26.7 | 20.0 | 24.8 |
| GLM-4 + AWR | 9B | 52.6 | 16.7 | 34.3 | 30.0 | 22.2 | 27.9 |
| GLM-4 + DigiRL | 9B | 63.2 | 30.0 | 34.3 | 26.7 | 26.7 | 31.5 |
| GLM-4 + WebRL | 9B | 57.9 | 50.0 | 48.6 | 36.7 | 37.8 | 43.0 |
| Llama3.1-8B Instruct | 8B | 0.0 | 3.3 | 2.9 | 3.3 | 11.1 | 4.8 |
| Llama3.1-8B + SFT | 8B | 36.8 | 6.7 | 20.0 | 33.3 | 17.8 | 20.6 |
| Llama3.1-8B + WebRL | 8B | 63.2 | 46.7 | 54.3 | 36.7 | 31.1 | 42.4 |
| Llama3.1-70B + SFT | 70B | 52.6 | 20.0 | 20.0 | 26.7 | 13.3 | 23.0 |
| Llama3.1-70B + WebRL | 70B | 78.9 | 50.0 | 54.3 | 40.0 | 44.4 | 49.1 |

  结论很清楚：WebRL 让 GLM-4-9B 从 6.1% 到 43.0%，让 Llama3.1-8B 从 4.8% 到 42.4%，让 Llama3.1-70B 达到 49.1%。它不仅超过同尺寸 SFT/Filtered BC/AWR/DigiRL，也超过 GPT-4-Turbo 和 GPT-4o 的直接 prompting 口径。

- **消融实验 (Ablation Studies)**：第一，去掉 replay buffer 后性能会随 phase 训练恶化，说明在线 curriculum 只看新数据会遗忘旧网页技能。第二，去掉 KL-constrained update 后，REINFORCE + value baseline 容易过拟合当前 phase，policy drift 更严重。第三，去掉 self-evolving curriculum，只用第一阶段生成任务，性能上升更慢且 ceiling 更低。第四，perplexity 过滤很关键：只复习太熟的数据 `[1,1/0.95]` 会退化，只学太难的数据 `[1/0.5,∞]` 也会退化，最佳区间是 `[1/0.95,1/0.5]`，对应 31.5% 的中间阶段表现。第五，ORM 自身评估也做了对比：8B ORM 在 WebArena-Lite test set / rollout set 上达到 80.8% / 79.4%，高于 GPT-4、Captioner+GPT-4 和 GPT-4V 的约 70%-73%。

| ORM / Evaluator | Test Dataset Accuracy | Rollout Accuracy |
|---|---:|---:|
| WebRL ORM 8B | 80.8 | 79.4 |
| GPT-4 | 71.9 | 71.2 |
| Captioner + GPT-4 | 72.6 | 73.3 |
| GPT-4V | 71.2 | 70.5 |

## 3. 图文占位符 (Visual Guides)
![系统架构图](./images/webrl_arch.png)

![Self-evolving curriculum 流程图](./images/webrl_curriculum.png)

![ORM 训练与评估示意图](./images/webrl_orm.png)

![KL-constrained policy update 图](./images/webrl_policy_update.png)

![WebArena-Lite 主结果表](./images/webrl_results.png)

## 4. 局限性与未来方向 (Limitations)
- WebRL 强依赖 WebArena-style HTML observation 和 element ID。它不是纯视觉 GUI agent，不能直接说明同一方法在 Android screenshot-only、Windows desktop 或 VSCode GUI 上也成立。
- ORM 是训练闭环的核心瓶颈。虽然论文中 8B ORM 比 GPT-4/GPT-4V prompt evaluator 更准，但 80% 左右 accuracy 仍然会产生误判，长期在线训练可能学习到 reward hacking。
- Curriculum 生成依赖 GPT-4o 过滤不可行任务，也依赖 critic score 控制难度；如果任务生成器偏了，训练分布会跟着偏。
- Replay buffer 只保存成功轨迹，能稳定训练但也可能忽略失败轨迹中有用的正确前缀。相比 DigiRL / Digi-Q 的 step-level value，它对失败轨迹的利用还不够细。
- WebArena interaction 成本高，真实网页会变，账号、网络、页面加载和反爬都会影响复现。未来方向是更强的 verifier、更通用的视觉+DOM 状态表示，以及能跨 Web/OS/Mobile 迁移的在线课程 RL。
