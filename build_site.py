#!/usr/bin/env python3
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "gui_agent_rl_survey.md"
SURVEY_URL = "https://arxiv.org/abs/2604.27955"


@dataclass
class Paper:
    number: int
    title: str
    link_text: str
    links: list[str]
    motivation: str
    method: str
    experiment: str
    tags: list[str]


METHOD_NOTES = {
    "DigiRL": [
        "训练不是直接从零在线探索，而是先把 AitW 等人类轨迹转成可执行的 device-control 经验，让模型学会基本动作格式和界面语义。",
        "在线阶段把 Android 任务参数化后并行运行，策略每一步看截图和指令，输出 tap/type/swipe/back 等动作；环境执行动作后返回新截图。",
        "奖励来自任务完成判定和 VLM evaluator：如果界面状态与目标语义一致，整条轨迹得到终局奖励；训练再用 advantage-weighted 更新把成功轨迹的关键动作权重放大。",
        "课程学习负责控制难度：过易任务不再占用采样，长期失败任务降低采样比例，模型主要在“刚好可学”的任务上迭代。",
    ],
    "Digi-Q": [
        "核心不是让策略在线试错，而是学一个动作价值函数 Q(s,a)：给定当前屏幕和候选动作，估计这个动作未来能否完成任务。",
        "VLM 主干大多冻结，只在中间表示上接轻量 Q head，离线 TD 学习用轨迹中的后续成功/失败信号回传到前面动作。",
        "推理时策略先生成多个候选动作，Q head 对候选做 rerank，选择长期价值最高的动作；这相当于用推理计算换掉昂贵的真实环境 rollout。",
        "这种方法适合真实手机、企业软件、账号系统等不能频繁试错的场景，因为训练只需要静态轨迹和可回放状态。",
    ],
    "Agent Q": [
        "Agent Q 把网页任务看成搜索树：节点是网页状态，边是点击、输入、选择等动作，目标是在树里找到能完成任务的路径。",
        "guided MCTS 用 value model 估计状态前景，优先扩展可能成功的分支；self-critique 对失败路径做解释，帮助构造更清楚的偏好对。",
        "训练不是只模仿成功轨迹，而是用 off-policy DPO：成功路径是 chosen，失败或低质量路径是 rejected，让模型学习为什么某些中间决策会把任务带偏。",
        "在线搜索可以在推理时继续使用，所以论文同时展示了“只靠训练后策略”和“训练后再加搜索”的两种能力边界。",
    ],
    "WebRL": [
        "WebRL 的关键在自演化课程：模型先做当前任务，失败后系统分析失败类型，再生成更适合当前能力的新任务，而不是固定训练集反复刷。",
        "Outcome-supervised reward model 只判断最终网页状态是否完成，不需要人工给每一步标注正确动作；这使得网页任务可以规模化在线采样。",
        "训练时维护 replay/filtering：过旧、过简单、明显错误的轨迹被过滤，近期成功和接近成功的轨迹保留，用来稳定策略更新。",
        "它解决的不是单个网页导航技巧，而是开放 LLM 在 WebArena 这类长任务中从低成功率逐步爬升的问题。",
    ],
    "UI-TARS": [
        "UI-TARS 把 GUI 交互统一成截图输入和动作 token 输出，不依赖某个平台独有 DOM，因此能覆盖桌面、网页和移动端。",
        "ARPO 在 GRPO 基础上加入 experience replay：如果当前 batch 全失败，训练仍可从 replay buffer 取成功轨迹形成有效梯度。",
        "任务选择器会筛掉 baseline 已经稳定解决的简单任务，也避免长期完全不可解任务拖慢训练；训练集中在稀疏奖励下仍有学习信号的任务。",
        "这篇的工程重点是把端到端 VLM policy optimization 真正接到多轮 GUI 环境，而不是只做静态 grounding。",
    ],
    "UI-TARS-2": [
        "UI-TARS-2 强化多轮训练：模型每步先产生 thought 或状态理解，再输出动作，训练目标覆盖整条交互序列。",
        "它把错误恢复纳入训练分布：真实多轮任务里前一步动作会改变 UI，模型必须根据新截图重新定位，而不能假设专家轨迹仍然成立。",
        "跨平台动作空间保持统一，平台差异交给环境 adapter；这使移动、桌面、网页任务可以共享同一个策略接口。",
        "论文更像系统技术报告，价值在于说明大模型 GUI agent 已经从单步定位走向 multi-turn RL 和长期上下文管理。",
    ],
    "GUI-R1": [
        "GUI-R1 借鉴 R1/RLVR 思路，把 GUI 动作格式设计成可规则验证：动作类型、坐标、文本参数都能被自动检查。",
        "训练数据很小但质量高，覆盖 mobile/desktop/web；每条样本都能产生格式奖励、动作类型奖励和坐标命中奖励。",
        "GRPO 不需要额外 critic，模型在同一状态下采样多个动作，用相对奖励区分哪一个更接近正确元素。",
        "最重要的现象是模型出现类似“先看布局再定位”的推理模式，说明 GUI 中只要奖励足够明确，不一定需要显式写满 reasoning 模板。",
    ],
    "UI-R1": [
        "UI-R1 研究的是 action prediction，而不是完整多轮任务：给截图和指令，模型需要输出正确动作与坐标。",
        "规则奖励分层检查：输出是否合法、动作类型是否对、坐标是否落在目标元素区域；这种细粒度 reward 避免只有最终对错的稀疏信号。",
        "训练样本只有百级，重点验证小数据 RL 是否能激活 Qwen2.5-VL 这类通用 VLM 的 GUI 定位能力。",
        "UI-R1-E 进一步把 grounding 效率作为目标，减少冗长推理和无效 token，让动作预测更适合在线 agent 调用。",
    ],
    "GUI-G2": [
        "GUI-G2 认为点击不是二值事件：目标元素附近的点也有一定价值，离中心越远价值越低，因此用二维高斯替代硬 IoU。",
        "奖励包含 point reward 和 coverage reward：前者评估预测点到元素中心的距离，后者评估预测分布与真实区域的重叠。",
        "元素越大，高斯方差越大；元素越小，奖励曲面越尖锐。这比固定阈值更符合不同 UI 控件的点击容忍度。",
        "它主要服务 GUI grounding RL，但思想可迁移到任何坐标式 GUI 动作，例如 tap、drag 起点、菜单项定位。",
    ],
    "SE-GUI": [
        "SE-GUI 从少量高质量 seed data 出发，不追求先堆大数据，而是让模型用自己的注意力和定位结果不断产生更有信息量的训练信号。",
        "dense policy gradient 把坐标误差转成连续反馈，让模型知道“差多少”，而不只是知道“对或错”。",
        "self-evolutionary RFT 用模型当前注意力图和预测行为筛选/修正样本，再进入下一轮训练，形成小数据自迭代。",
        "它适合专业软件和高分辨率界面，因为这类场景控件密集、纯 SFT 泛化差、人工标注成本高。",
    ],
    "InfiGUI-G1": [
        "InfiGUI-G1 关注 grounding 中的探索效率：同一截图里可能有多个相似元素，模型需要探索候选而不是只生成一个点。",
        "AEPO 先生成多个答案扩大搜索，再根据 useful/cost 的效率给探索奖励；无效的长推理或重复候选会被压低权重。",
        "论文特别指出强制 CoT 未必总是好事：在坐标任务里，冗长语言推理可能挤占视觉定位能力，因此奖励要同时看语义和定位效率。",
        "最终目标是让模型学会在相似元素中找到功能正确的那个，而不只是点中外观相似的区域。",
    ],
    "UI-AGILE": [
        "UI-AGILE 把问题拆成训练和推理两端：训练端解决稀疏奖励，推理端解决高分辨率小控件定位。",
        "训练端用连续奖励和 Simple Thinking reward，鼓励简洁但必要的视觉推理；cropping-based resampling 让模型多看难定位区域。",
        "推理端先把大图分块，选出可能区域，再在局部做精确 grounding，避免整图缩放后小文字和小按钮丢失。",
        "这篇说明 GUI grounding 不能只靠训练算法，输入分辨率、裁剪策略和候选区域选择同样决定上限。",
    ],
    "InfiGUI-R1": [
        "InfiGUI-R1 的目标是把 reactive actor 变成 deliberative reasoner：模型不是直接点，而是先形成空间关系和子目标。",
        "Actor2Reasoner 第一阶段注入 reasoning：用 teacher 的跨模态空间推理教模型理解“左上角第二个按钮”“表格第三列”等关系。",
        "第二阶段用 RL 强化 deliberation：在有错误恢复、子目标选择和复杂布局的任务里，用奖励让模型保留有用推理、抑制空泛推理。",
        "它对应 GUI agent 的 System-2 路线：不是所有任务都需要长推理，但复杂界面需要可控的中间思考。",
    ],
    "BacktrackAgent": [
        "BacktrackAgent 把每一步执行后的页面当成新的判断对象：动作不是发出去就结束，而要检查后果是否符合预期。",
        "verifier 负责看状态是否前进，judger 判断是否需要回退，reflector 根据错误原因生成修正动作。",
        "训练数据包含动作后 outcome page，因此模型学到的是“执行-检查-回退-重试”的闭环，而不是单步点击分类。",
        "它特别适合长任务，因为 GUI 任务一旦误入错误页面，如果没有回退机制，后续动作再准也无法完成目标。",
    ],
    "VSC-RL": [
        "VSC-RL 把长任务拆成可学习的 subgoal：先让 VLM 或 planner 生成中间目标，再训练策略在每个子目标条件下行动。",
        "SubGoal Evidence Lower Bound 把最终成功、子目标可达性和与参考策略的偏差放进同一个优化目标。",
        "这样做的好处是 reward 不必从终局一次性回传几十步，模型可以在“到达某个中间界面”时得到更短程的学习信号。",
        "它是层级 RL 在 GUI/Web/Mobile 任务中的自然落地，尤其适合搜索、填写、确认这种多阶段流程。",
    ],
    "Explorer": [
        "Explorer 先让 agent 在真实网页中探索，而不是等人工写任务；它记录可点击元素、页面跳转、表单结果和截图。",
        "当探索发现一条可完成的路径后，annotator 反向生成自然语言指令，相当于从行为轨迹合成任务描述。",
        "筛选器再剔除不可复现、语义不清或重复的轨迹，得到大规模多模态 web trajectory 数据。",
        "这条路线的本质是把环境探索变成数据生成器，为后续 SFT、offline RL 和 online curriculum 提供便宜轨迹。",
    ],
    "UI-S1": [
        "UI-S1 介于离线和在线之间：训练时不真正访问环境，而是在离线轨迹上让当前模型产生偏离，再模拟如何修回专家路径。",
        "Patch Module 是关键，它负责把模型动作导致的 OOD 状态补回可训练状态，避免离线 rollout 一偏离就无数据可学。",
        "奖励同时看未来折扣回报、step advantage 和 episode advantage，让模型知道单步动作和整条轨迹之间的关系。",
        "SOP 指标用来估计 semi-online 训练质量，帮助判断模型是否真的学到动态修正能力。",
    ],
    "Hi-Agent": [
        "Hi-Agent 采用高低层分工：高层 reasoning model 负责理解任务、拆子目标和决定下一阶段，低层 action model 负责把子目标变成具体手机动作。",
        "高层输出不是最终坐标，而是语义计划，例如进入某 app、找到某设置项、确认某状态；低层再用当前截图执行 tap/type/swipe。",
        "训练时既需要示范轨迹让两层对齐，也需要 RL 奖励高低层协同：高层目标如果不可执行，低层会失败；低层误操作也会破坏高层计划。",
        "它代表移动 GUI 的层级控制路线，用结构分工降低长任务 credit assignment 难度。",
    ],
    "MobileRL": [
        "MobileRL 面向真实移动 GUI online RL，核心难点是任务难度重尾和 emulator 采样慢。",
        "Difficulty-Adaptive GRPO 根据历史成功率给任务调权：太简单的少训，刚好有学习信号的多训，长期失败的暂时过滤。",
        "positive replay 保存困难任务里的成功轨迹，避免成功样本很少时被新 batch 的失败淹没；Shortest-Path Reward Adjustment 惩罚绕路。",
        "系统上需要大规模 Android emulator 并行，CPU/内存/ADB I/O 往往比 GPU 更先成为瓶颈。",
    ],
    "WebAgent-R1": [
        "WebAgent-R1 把优化单位从单步动作改成完整多轮 trajectory：只有整条轨迹完成任务，前面的延迟动作才得到正向 credit。",
        "Multi-Turn GRPO 在同一任务上采样多条轨迹，用终局和过程信号比较轨迹优劣，再更新整段动作序列。",
        "dynamic context compression 让 agent 每步写 observation summary，压缩历史页面和操作，避免长网页任务超出上下文。",
        "这篇的核心启发是 web agent 不能只优化下一步点击，必须训练“为了后面成功而暂时执行看似无收益步骤”的能力。",
    ],
    "DART-GUI": [
        "DART-GUI 重点不是新奖励，而是解决 GUI RL 的系统吞吐：环境 rollout、模型推理、数据管理、训练被拆成异步模块。",
        "environment cluster 持续运行 GUI 任务，rollout service 调用当前策略，data manager 管理轨迹和难度，trainer 消费高价值样本。",
        "adaptive data curation 会给困难任务更多 rollout 和更长 horizon，给高熵步骤更高训练优先级，同时用 importance sampling 修正旧策略轨迹偏差。",
        "它说明在线 GUI RL 的瓶颈常在架构调度：没有异步和筛选，再好的算法也会被浏览器/VM/模拟器 I/O 拖住。",
    ],
    "Mano": [
        "Mano 使用三阶段训练：SFT 先学动作格式，offline RL 从已有轨迹中学偏好和价值，online RL 再在环境里闭环修正。",
        "高保真模拟环境用于产生更接近真实 GUI 的状态变化，verification module 则检查执行后是否真的前进。",
        "reward 不是单一终局信号，而是结合任务完成、动作合理性、状态进展和错误恢复的综合奖励。",
        "它代表模型报告型工作：把数据、模拟、验证和 RL pipeline 组合成面向真实 GUI 的完整系统。",
    ],
    "GELab": [
        "GELab-Zero 更像环境与训练平台：先构建可自动重置、可验证、可探索的 GUI 环境，再让 agent 零人工或低人工采样。",
        "任务通过环境状态定义完成条件，而不是要求人工逐步标注；agent 在环境里探索并积累成功/失败经验。",
        "它支持移动端 self-evolution：模型一边执行，一边用环境反馈筛选新轨迹，再用于下一轮训练。",
        "这类平台的价值是把 GUI RL 从单篇算法推进到可复现基础设施。",
    ],
    "Step-GUI": [
        "Step-GUI 针对长任务稀疏奖励提出 step-level reward：不只问最后成功没有，还问每一步是否让任务更接近完成。",
        "自动 reward annotation 用模型/规则判断当前步骤的进展，成本远低于人工逐步标注。",
        "训练时 process reward 与 final reward 结合，避免模型学会刷终局判定或在中间无意义徘徊。",
        "它适合 AndroidWorld、OSWorld 这类多步任务，因为这些环境里很多正确动作短期看不到终局收益。",
    ],
    "MAI-UI": [
        "MAI-UI 是跨平台 GUI agent 系统，强调从 mobile/web/desktop 多环境收集轨迹，并统一到同一动作接口。",
        "训练通常包含 SFT 和 RL 两段：前者建立基础可控性，后者利用环境或 verifier 反馈提升多轮成功率。",
        "系统层面需要大量并行环境 worker、统一 observation schema、轨迹缓冲和异步权重同步。",
        "它的价值在工程化：说明 GUI agent 不能只在单个 benchmark 上写 prompt，而要形成跨生态训练管线。",
    ],
    "GUI-Eyes": [
        "GUI-Eyes 认为很多失败不是策略问题，而是看不清：小图标、小文字、密集控件导致 VLM 视觉输入不够。",
        "它引入 tool-augmented perception，让模型可以请求放大、裁剪、局部重看或辅助定位，而不是一次性从全图猜坐标。",
        "训练用少量样本让模型学会何时调用视觉工具、如何把工具结果转成动作预测。",
        "这条路线把“主动感知”纳入 GUI agent，和只靠更大模型或更长 CoT 的方案形成互补。",
    ],
    "DynaWeb": [
        "DynaWeb 学一个网页世界模型：输入当前网页状态和动作，预测下一状态或可用观察，让 agent 能在模型里练习。",
        "真实网页 rollout 成本高且慢，世界模型中的 dream rollout 可以快速生成候选轨迹，再用真实环境少量校准。",
        "训练策略时可先在模拟网页中搜索或 RL，再把高价值轨迹迁回真实环境验证。",
        "关键风险是 sim-to-real gap：世界模型预测错的页面状态会诱导策略学到现实中不可执行的路径。",
    ],
    "UltraCUA": [
        "UltraCUA 不把 computer use 限定为鼠标键盘，而是把 GUI 动作和 API/tool call 放进同一个 hybrid action space。",
        "能用 API 的地方直接调用结构化工具，不能用 API 的长尾软件再回退到截图、点击和键盘输入。",
        "训练时模型要学会选择动作通道：API 更可靠但覆盖有限，GUI 更通用但慢且易错。",
        "这很接近真实部署，因为真实用户的软件生态既有浏览器/文件/系统 API，也有完全不可编程的图形界面。",
    ],
    "ComputerRL": [
        "ComputerRL 把成千上万个虚拟桌面环境接入在线 RL，核心是分布式 rollout 基础设施。",
        "API-GUI 范式允许 agent 在桌面任务中混用结构化系统调用和视觉 GUI 操作，既提升效率又保留长尾覆盖。",
        "系统把虚拟机/远程桌面、模型推理、奖励检查和训练解耦，避免任何单个环节阻塞整体吞吐。",
        "它展示了 OSWorld 等桌面任务上开源 CUA 的上限来自模型能力和环境并行规模共同作用。",
    ],
    "ZeroGUI": [
        "ZeroGUI 的目标是把人工标注成本降到接近零：任务、执行和验证尽量由环境或模型自动完成。",
        "在线学习中 agent 生成轨迹，自动 verifier 判断成功与否，失败轨迹用于构造反例或改写课程。",
        "它更关注数据闭环：如何从零人工示范出发不断产生可训练样本，而不是依赖大规模专家点击。",
        "主要风险是 verifier 错误和 reward hacking，所以需要多重检查或保守过滤。",
    ],
    "ProgRM": [
        "ProgRM 学的是 progress reward：当前状态相比上一步是否更接近目标，而不是只有终局 success/fail。",
        "reward model 需要理解任务语义、当前页面和历史动作，判断进展、停滞或倒退。",
        "在长任务中，progress reward 能把几十步后的终局信号切成更密的训练反馈，降低 credit assignment 难度。",
        "它适合和 GRPO/PPO 或轨迹筛选结合，用来挑选更有效的中间步骤。",
    ],
    "Co-EPG": [
        "Co-EPG 同时优化 planning 和 grounding：planner 产生高层步骤，grounder 把步骤落到具体 UI 元素。",
        "两者 co-evolution 的意思是 planner 会根据 grounder 的实际可执行能力调整计划，grounder 也从 planner 的更好任务分解中学习。",
        "如果只训练 planner，计划可能不可点；只训练 grounder，模型可能局部动作准但全局方向错。",
        "这篇代表 GUI agent 中“语义规划”和“视觉落地”必须一起进化的方向。",
    ],
    "AEBPO": [
        "AEBPO 关注多轮 agent RL 的 entropy collapse：策略训练后过早收敛到少数动作，探索能力下降。",
        "Entropy-balanced 的思想是在不同阶段动态调节探索强度：既不能一直高熵乱试，也不能太早低熵僵化。",
        "它虽是更通用的 agent RL 算法，但对 Web/GUI 任务重要，因为长任务早期需要探索，后期需要稳定执行。",
        "论文结果说明少量 RL samples 如果配合好的熵控制，也能显著提升工具使用和网页任务能力。",
    ],
    "Nested Browser": [
        "NestBrowse 把浏览任务分成外层 agent 和内层 browser-use：外层负责问题分解和信息整合，内层负责页面探索。",
        "内层动作空间保持 minimal and complete，避免 ReAct 工具调用越堆越复杂。",
        "这种嵌套结构能处理深层网页信息查找：外层不必记住每个 DOM 细节，内层专注局部导航。",
        "它对 GUI RL 的启发是：复杂任务可通过结构化子代理降低动作空间和上下文压力。",
    ],
    "WebSynthesis": [
        "WebSynthesis 先训练网页世界模型，再用 MCTS 在世界模型中搜索高质量轨迹。",
        "MCTS 的每个节点对应虚拟网页状态，边对应动作；搜索能回溯并比较不同路径，比随机探索更高效。",
        "生成的轨迹经过筛选后用于 SFT/RL，目标是用低成本 synthetic trajectories 替代大量真实网页交互。",
        "它把 Agent Q 的搜索思想和 DynaWeb 的世界模型思想结合起来。",
    ],
    "WebWorld": [
        "WebWorld 试图把开放网页交互训练成大规模世界模型，输入动作后预测网页后续状态、可用元素和任务进展。",
        "与单站点模拟器不同，它覆盖大量 open-web interactions，目标是让模型学到网页交互的一般规律。",
        "下游可以用它生成训练轨迹，也可以在推理时做 lookahead search，先在模型里试几步再真实执行。",
        "它代表 GUI RL 向“agent-native environment model”迁移：先学数字世界，再训练数字居民。",
    ],
    "Code2World": [
        "Code2World 不直接预测像素，而是预测可渲染代码，例如 HTML/CSS/布局结构，再渲染成下一 UI。",
        "这样做的优势是状态可解释、可编辑、可验证；如果渲染结果不对，还可以用视觉反馈修正代码。",
        "Render-Aware RL 用渲染后的视觉一致性和动作一致性作为 reward，让模型学会生成既像界面又符合交互后果的代码。",
        "它把 GUI 世界模型从黑箱图像生成推进到结构化生成，降低后续验证和轨迹合成成本。",
    ],
    "AgentCPM": [
        "AgentCPM-Explore 面向小模型深度探索，目标是在有限参数下学会长任务工具使用和网页/GUI 决策。",
        "parameter-space fusion 缓解 SFT 后的能力遗忘，reward denoising 降低环境反馈噪声，context refinement 压缩长历史。",
        "训练强调 deep exploration：让模型在任务空间里找到成功路径，再用成功/失败信号改进策略。",
        "它说明 GUI/Web agent 并不一定只能靠超大模型，小模型也能通过更好的探索和奖励清洗获得竞争力。",
    ],
    "ClawGUI": [
        "ClawGUI 是全栈框架：训练、评测、部署分别由 ClawGUI-RL、ClawGUI-Eval、ClawGUI-Agent 承接。",
        "训练端支持并行虚拟环境和真实设备，奖励端包含 Process Reward Model，部署端支持多平台和聊天入口。",
        "统一 action schema 和 environment adapter 是关键，否则不同 benchmark、设备和模型无法复用轨迹和评测结果。",
        "它更像 GUI agent 生态标准化工作，回答“怎么把论文算法变成可复现平台”。",
    ],
    "AgentTrek": [
        "AgentTrek 利用网页教程生成任务：教程天然包含目标、步骤和预期结果，是低成本轨迹来源。",
        "管线先抓取 tutorial-like 文本，再转成结构化任务和步骤，让 VLM agent 在网页中执行，最后由 evaluator 验证。",
        "它同时生成 HTML/function-call 轨迹和 screenshot/pixel-action 轨迹，能服务不同类型的 web agent。",
        "相比人工标注，它的关键优势是可扩展；相比纯随机探索，它的任务语义更清楚。",
    ],
    "OS-Copilot": [
        "OS-Copilot/FRIDAY 把 agent 放到真实操作系统，任务覆盖浏览器、文件、终端、Office 和第三方应用。",
        "FRIDAY 会积累技能：成功完成的任务被抽象成可复用程序或操作模式，下次遇到相似任务时直接调用。",
        "它不是严格的 RL 论文，但提供了 GUI/CUA 自我改进的重要系统视角：长期记忆和技能库可以减少重复探索。",
        "对 GUI RL 的启示是，未来 agent 不能只靠单次 episode 学习，还要在跨任务、跨天的使用中持续积累技能。",
    ],
}


CATEGORY_RULES = [
    ("Online RL", ["DigiRL", "WebRL", "MobileRL", "WebAgent-R1", "DART-GUI", "ComputerRL", "ZeroGUI", "AgentCPM"]),
    ("Offline RL", ["Digi-Q", "Agent Q", "UI-R1", "GUI-R1", "UI-TARS", "ARPO", "AEBPO"]),
    ("Grounding", ["GUI-G2", "SE-GUI", "InfiGUI-G1", "UI-AGILE", "GUI-Eyes", "InfiGUI-R1"]),
    ("Hybrid", ["UI-S1", "Hi-Agent", "UltraCUA", "Co-EPG", "Mano", "ClawGUI", "BacktrackAgent", "VSC-RL"]),
    ("World Model", ["DynaWeb", "WebSynthesis", "WebWorld", "Code2World"]),
    ("Data", ["Explorer", "AgentTrek", "GELab", "Step-GUI", "MAI-UI", "NestBrowse"]),
    ("Reward", ["ProgRM", "ZeroGUI", "GUI-G2", "Step-GUI"]),
    ("OS/Desktop", ["OS-Copilot", "ComputerRL", "UltraCUA", "DART-GUI"]),
    ("Mobile", ["DigiRL", "Digi-Q", "MobileRL", "Hi-Agent", "GELab", "MAI-UI"]),
    ("Web", ["WebRL", "Agent Q", "WebAgent-R1", "DynaWeb", "WebSynthesis", "WebWorld", "Explorer", "NestBrowse", "AgentTrek"]),
]


def inline_md(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<a href="\2" target="_blank" rel="noreferrer">\1</a>',
        text,
    )
    text = re.sub(
        r"(?<![\"'=])(https?://[^\s，。；、)]+)",
        r'<a href="\1" target="_blank" rel="noreferrer">\1</a>',
        text,
    )
    return text


def block_md(lines: list[str]) -> str:
    out: list[str] = []
    buf: list[str] = []

    def flush_paragraph() -> None:
        if buf:
            out.append(f"<p>{inline_md(' '.join(buf))}</p>")
            buf.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_paragraph()
            continue
        buf.append(line)
    flush_paragraph()

    # Second pass for simple unordered lists, preserving paragraphs above for non-list sections.
    if any(line.strip().startswith("- ") or re.match(r"^\d+\.\s+", line.strip()) for line in lines):
        out = []
        list_type: str | None = None
        para: list[str] = []

        def close_list() -> None:
            nonlocal list_type
            if list_type:
                out.append(f"</{list_type}>")
                list_type = None

        for raw in lines:
            line = raw.strip()
            if not line:
                if para:
                    out.append(f"<p>{inline_md(' '.join(para))}</p>")
                    para.clear()
                close_list()
                continue
            if line.startswith("- "):
                if para:
                    out.append(f"<p>{inline_md(' '.join(para))}</p>")
                    para.clear()
                if list_type != "ul":
                    close_list()
                    out.append("<ul>")
                    list_type = "ul"
                out.append(f"<li>{inline_md(line[2:])}</li>")
            elif re.match(r"^\d+\.\s+", line):
                if para:
                    out.append(f"<p>{inline_md(' '.join(para))}</p>")
                    para.clear()
                if list_type != "ol":
                    close_list()
                    out.append("<ol>")
                    list_type = "ol"
                item_text = re.sub(r"^\d+\.\s+", "", line)
                out.append(f"<li>{inline_md(item_text)}</li>")
            else:
                close_list()
                para.append(line)
        if para:
            out.append(f"<p>{inline_md(' '.join(para))}</p>")
        close_list()
    return "\n".join(out)


def extract_between(text: str, start: str, end: str | None = None) -> str:
    start_idx = text.index(start) + len(start)
    end_idx = text.index(end, start_idx) if end else len(text)
    return text[start_idx:end_idx].strip()


def tags_for(title: str) -> list[str]:
    tags: list[str] = []
    for tag, needles in CATEGORY_RULES:
        if any(needle.lower() in title.lower() for needle in needles):
            tags.append(tag)
    return tags or ["GUI RL"]


def parse_papers(section: str) -> list[Paper]:
    chunks = re.split(r"(?=^###\s+\d+\.\s+)", section, flags=re.M)
    papers: list[Paper] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^###\s+(\d+)\.\s+(.+)$", chunk, flags=re.M)
        if not m:
            continue
        number = int(m.group(1))
        title = m.group(2).strip()
        lines = chunk.splitlines()[1:]
        body = "\n".join(lines).strip()
        link_match = re.search(r"^链接：(.+)$", body, flags=re.M)
        link_text = link_match.group(1).strip() if link_match else ""
        links = re.findall(r"https?://[^\s，。；、]+", link_text)
        fields: dict[str, str] = {}
        for label, next_label in [("动机", "方案"), ("方案", "实验"), ("实验", None)]:
            pattern = rf"{label}：(.*?)(?=\n\n{next_label}：|\Z)" if next_label else rf"{label}：(.*)"
            fm = re.search(pattern, body, flags=re.S)
            fields[label] = re.sub(r"\s+", " ", fm.group(1)).strip() if fm else ""
        papers.append(
            Paper(
                number=number,
                title=title,
                link_text=link_text,
                links=links,
                motivation=fields["动机"],
                method=fields["方案"],
                experiment=fields["实验"],
                tags=tags_for(title),
            )
        )
    return papers


def parse_heading_sections(section: str) -> list[tuple[str, str]]:
    chunks = re.split(r"(?=^###\s+)", section, flags=re.M)
    parsed: list[tuple[str, str]] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        title = re.sub(r"^###\s+", "", lines[0]).strip()
        parsed.append((title, block_md(lines[1:])))
    return parsed


def parse_references(section: str) -> list[tuple[str, str]]:
    refs = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        item = line[2:]
        match = re.match(r"^(.+):\s+(https?://\S+)$", item)
        if match:
            name, url = match.groups()
            refs.append((name, url))
    return refs


def slug_for(paper: Paper) -> str:
    raw = paper.title.lower()
    raw = raw.replace("π", "pi").replace("τ", "tau")
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return f"{paper.number:02d}-{slug[:64].strip('-')}"


def detail_notes_for(paper: Paper) -> list[str]:
    for key in sorted(METHOD_NOTES, key=len, reverse=True):
        notes = METHOD_NOTES[key]
        if key.lower() in paper.title.lower():
            return notes
    return [
        f"这篇论文的核心对象是 {paper.tags[0]} 场景下的 GUI agent，先把任务转成可观测状态、可执行动作和可验证反馈三部分。",
        "方法上先保证动作空间可执行，再把环境反馈转成训练信号，避免只在静态示范上做行为克隆。",
        "如果它属于数据或系统论文，重点就不是单个 reward 公式，而是如何让轨迹采样、筛选、验证和训练形成闭环。",
        "读这篇时应重点关注它的 observation/action/reward 三元组，以及它如何处理长时序、稀疏奖励和环境 I/O 成本。",
    ]


def task_form_for(paper: Paper) -> list[tuple[str, str]]:
    title = paper.title.lower()
    tags = set(paper.tags)
    if "World Model" in tags:
        return [
            ("观察", "当前 UI 状态、历史动作和任务指令；状态可以是截图、DOM、可访问性树或可渲染代码。"),
            ("动作", "真实或模拟的 click/type/scroll/API call；世界模型学习动作之后界面如何变化。"),
            ("奖励", "由预测状态是否支持完成任务、真实环境校准结果或后续轨迹成功率给出。"),
        ]
    if "Grounding" in tags:
        return [
            ("观察", "屏幕截图、目标文本或自然语言指令，有时还包含裁剪区域和候选元素。"),
            ("动作", "输出坐标、边界框或带坐标的 GUI action，例如 click(x,y)。"),
            ("奖励", "坐标命中、IoU、距离衰减、高斯覆盖度或语义定位是否正确。"),
        ]
    if "Mobile" in tags or "android" in title:
        return [
            ("观察", "Android 当前截图、任务指令，部分系统还读 accessibility tree、ADB 状态或应用数据库。"),
            ("动作", "tap、type、swipe、scroll、back/home、finish 等移动端原子动作。"),
            ("奖励", "任务完成由 ADB/SQLite/文件状态、页面状态或 VLM judge 验证；中间奖励看状态进展和路径长度。"),
        ]
    if "OS/Desktop" in tags or "computer" in title or "os-" in title:
        return [
            ("观察", "桌面截图、窗口状态、文件系统或应用状态，通常通过 VM/VNC/系统 API 获取。"),
            ("动作", "鼠标点击、键盘输入、拖拽、快捷键，以及部分 API/MCP/tool call。"),
            ("奖励", "检查文件、数据库、应用 UI、终端输出或系统状态是否达到任务要求。"),
        ]
    if "Web" in tags or "browser" in title:
        return [
            ("观察", "网页截图、URL、DOM/accessibility tree、任务指令和历史操作摘要。"),
            ("动作", "click、type、select、scroll、navigate、submit，有的系统还允许 browser tool/API。"),
            ("奖励", "目标 URL、DOM 变化、后端数据库状态、表单提交结果或 reward model 判断。"),
        ]
    return [
        ("观察", "当前 GUI 视觉状态、任务指令和必要的历史上下文。"),
        ("动作", "统一 GUI action schema，包括 click/type/scroll/swipe/back/finish 或工具调用。"),
        ("奖励", "由终局完成、过程进展、动作合法性、坐标定位或 verifier 判断组成。"),
    ]


def training_recipe_for(paper: Paper) -> list[str]:
    tags = set(paper.tags)
    title = paper.title.lower()
    recipe = []
    if "Offline RL" in tags:
        recipe.append("离线阶段先从静态轨迹、成功/失败对或规则可验证样本中学习，优点是便宜、安全、可复现。")
        recipe.append("优化通常用 DPO、offline GRPO、Q-learning、IQL/CQL 或 value reranking，重点是避免策略偏离数据分布后无人纠正。")
    if "Online RL" in tags:
        recipe.append("在线阶段让当前策略真实或半真实地执行任务，把环境返回的成功/失败和过程状态写入 replay buffer。")
        recipe.append("训练一般需要课程学习、positive replay、任务过滤和异步 rollout，否则大量失败轨迹会吞掉训练信号。")
    if "Hybrid" in tags:
        recipe.append("混合路线先用 SFT/offline 数据冷启动，再用在线反馈、分层规划、API-GUI 混合动作或 verifier 做二次强化。")
        recipe.append("它的核心取舍是：离线数据给稳定性，在线环境给真实反馈，系统结构负责把两者接起来。")
    if "Grounding" in tags:
        recipe.append("Grounding 训练必须把视觉空间误差转成可优化 reward；常见做法包括距离奖励、IoU、高斯奖励和裁剪重采样。")
    if "World Model" in tags:
        recipe.append("世界模型路线先学习环境动力学，再在模型中做 dream rollout、MCTS 或轨迹合成，最后回到真实环境校准。")
    if "Reward" in tags:
        recipe.append("奖励工程关注从 binary success 扩展到 continuous/process/composite reward，减少长任务中的零梯度问题。")
    if "Data" in tags:
        recipe.append("数据路线把网页教程、环境探索、模拟器或世界模型变成轨迹工厂，关键是自动筛选可复现成功轨迹。")
    if not recipe:
        recipe.append("训练流程一般从动作格式 SFT 开始，再引入可验证反馈做策略优化，最后在动态环境中验证多轮成功率。")
    if "grpo" in paper.method.lower() or "r1" in title:
        recipe.append("如果使用 GRPO/RLVR，模型会在同一状态采样多个候选动作，用相对奖励更新策略，避免训练单独 critic。")
    return recipe


def innovations_for(paper: Paper) -> list[str]:
    tags = set(paper.tags)
    title = paper.title.lower()
    items = []
    if "r1" in title or "grpo" in paper.method.lower():
        items.append("把 GUI 任务改造成可验证奖励问题，让策略优化不必依赖昂贵的逐步人工标注。")
    if "Online RL" in tags:
        items.append("把环境交互纳入训练闭环，直接优化真实多轮成功率，而不是只优化离线动作匹配。")
    if "Offline RL" in tags:
        items.append("在不访问真实环境的前提下利用轨迹、偏好或 value signal，降低安全风险和交互成本。")
    if "Grounding" in tags:
        items.append("把视觉定位从硬命中扩展到连续或主动感知反馈，使坐标错误能产生可学习梯度。")
    if "World Model" in tags:
        items.append("用可学习环境模型替代部分真实 GUI rollout，把慢速 I/O 变成高速模拟采样。")
    if "Hybrid" in tags:
        items.append("通过层级、验证器、工具/API 或半在线机制，把长任务拆成更可控的子问题。")
    if "Data" in tags:
        items.append("把探索、教程、模拟器或世界模型变成轨迹生成器，减少人工收集 demonstration 的依赖。")
    if not items:
        items.append("贡献点主要在把 GUI agent 的训练、评测或部署流程标准化，让后续算法能复用同一套接口。")
    return items[:4]


def implementation_checklist_for(paper: Paper) -> list[str]:
    tags = set(paper.tags)
    checklist = [
        "先固定 action schema，并写一个能把模型输出解析成真实 GUI 操作的 adapter。",
        "准备 verifier：至少要能判断终局成功；如果是 grounding 论文，还要能计算坐标/区域奖励。",
    ]
    if "Web" in tags:
        checklist.append("Web 任务优先用 Playwright/BrowserGym/WebArena 这类可重置环境，记录 URL、DOM、截图和后端状态。")
    if "Mobile" in tags:
        checklist.append("Mobile 任务需要 Android Emulator/ADB、snapshot reset、任务参数化和状态检查脚本。")
    if "OS/Desktop" in tags:
        checklist.append("Desktop 任务需要 VM/VNC/截图通道，并把文件系统或应用状态检查写成可重复 verifier。")
    if "Online RL" in tags:
        checklist.append("在线训练要异步化 rollout 与 learner，并维护 replay/filtering，否则失败轨迹会占满 batch。")
    if "Offline RL" in tags:
        checklist.append("离线训练要检查轨迹覆盖和负样本质量，避免模型学到只在数据集状态里有效的捷径。")
    if "World Model" in tags:
        checklist.append("世界模型必须保留真实环境校准集，持续检查 sim-to-real gap，而不是只看模拟成功率。")
    if "Grounding" in tags:
        checklist.append("Grounding 实验要保留原始分辨率、元素框和坐标归一化规则，否则不同 benchmark 不可比。")
    return checklist[:6]


def primary_environment(paper: Paper) -> str:
    tags = set(paper.tags)
    title = paper.title.lower()
    if "Mobile" in tags or "android" in title:
        return "Mobile / Android emulator / ADB"
    if "Web" in tags or "browser" in title:
        return "Web / Browser / Playwright"
    if "OS/Desktop" in tags or "computer" in title or "os-" in title:
        return "Desktop / OS / VM / VNC"
    if "World Model" in tags:
        return "Simulated / learned world model"
    return "Cross-platform GUI"


def action_space_summary(paper: Paper) -> str:
    tags = set(paper.tags)
    if "Grounding" in tags:
        return "坐标、目标框、click(x,y) 或局部裁剪定位"
    if "World Model" in tags:
        return "真实 GUI action + 世界模型中的虚拟 action transition"
    if "OS/Desktop" in tags:
        return "mouse / keyboard / drag / hotkey / API or MCP tool"
    if "Mobile" in tags:
        return "tap / type / swipe / scroll / back / home / finish"
    if "Web" in tags:
        return "click / type / select / scroll / navigate / submit"
    return "统一 GUI action schema"


def reward_summary(paper: Paper) -> str:
    tags = set(paper.tags)
    method = (paper.method + " " + paper.title).lower()
    if "progress" in method or "progrm" in method:
        return "progress/process reward + final success"
    if "gaussian" in method or "g2" in method:
        return "Gaussian continuous grounding reward"
    if "Grounding" in tags:
        return "坐标命中、距离、IoU、coverage 或语义匹配"
    if "World Model" in tags:
        return "world-model rollout 成功率、真实环境校准和轨迹可执行性"
    if "Online RL" in tags:
        return "环境终局成功、verifier/VLM judge、replay 中的成功轨迹"
    if "Offline RL" in tags:
        return "离线轨迹回报、偏好对、Q value 或规则标签"
    return "任务完成 + 过程进展 + verifier"


def optimization_summary(paper: Paper) -> str:
    text = (paper.title + " " + paper.method).lower()
    if "grpo" in text:
        return "GRPO / RLVR"
    if "dpo" in text:
        return "DPO / preference optimization"
    if "q-value" in text or " q " in text or "q 函数" in text:
        return "offline TD / Q-value reranking"
    if "mcts" in text:
        return "MCTS / tree search"
    if "world model" in text or "dyna" in text:
        return "model-based RL / synthetic rollout"
    if "curriculum" in text:
        return "online curriculum RL"
    return "SFT warmup + RL/RFT 或系统级闭环优化"


def confidence_note(paper: Paper) -> str:
    if paper.links:
        return "⚠️ 数值主要按论文/技术报告/综述自报口径整理，未声明为第三方独立复现。"
    return "待核：综述列出该工作但未提供稳定 arXiv 链接或完整公开页，数值只按综述摘要口径保留。"


def profile_rows(paper: Paper) -> list[tuple[str, str]]:
    return [
        ("路线", " / ".join(paper.tags)),
        ("环境", primary_environment(paper)),
        ("动作空间", action_space_summary(paper)),
        ("奖励信号", reward_summary(paper)),
        ("优化方式", optimization_summary(paper)),
        ("读表口径", confidence_note(paper)),
    ]


def pipeline_for(paper: Paper) -> list[str]:
    tags = set(paper.tags)
    if "World Model" in tags:
        return [
            "收集真实 GUI 轨迹或网页交互日志，得到 state-action-next state 三元组。",
            "训练 action-conditioned world model，让模型预测动作后的下一界面、DOM、截图或可渲染代码。",
            "在世界模型里做 dream rollout / MCTS / synthetic trajectory generation，低成本探索多条路径。",
            "把模拟中筛出的高质量轨迹用于策略微调，再用少量真实环境校准 sim-to-real gap。",
        ]
    if "Grounding" in tags:
        return [
            "输入截图和目标描述，模型产生一个或多个坐标/框/局部候选。",
            "verifier 根据目标元素框、距离、高斯覆盖或语义匹配计算连续奖励。",
            "策略优化放大命中目标、语义正确、成本较低的候选，压低乱点、冗长推理和重复探索。",
            "推理时可结合裁剪、重采样、视觉工具或候选 rerank，提高小控件和密集界面定位。",
        ]
    if "Online RL" in tags:
        return [
            "任务采样器选择一批当前模型有机会学会的 GUI/Web/Mobile 任务。",
            "rollout worker 在浏览器、VM 或 Android emulator 中执行策略，记录截图、动作、状态和终局结果。",
            "verifier / reward model / 环境脚本把轨迹转成成功、失败或过程进展信号。",
            "trainer 用 GRPO/AWR/PPO/自定义策略优化更新模型，同时 replay/filtering 保留稀有成功轨迹。",
        ]
    if "Offline RL" in tags:
        return [
            "从示范轨迹、失败轨迹、候选动作或静态 benchmark 中构造可训练样本。",
            "把样本转成偏好对、Q-learning transition、规则奖励标签或 offline GRPO group。",
            "训练策略、value/Q head 或 reranker，使模型在不访问环境时也能区分好坏动作。",
            "推理时用训练后的策略直接行动，或生成多个候选后由 value/reward 模型重排。",
        ]
    if "Hybrid" in tags:
        return [
            "先用 SFT 或离线轨迹让模型学会基本动作格式和界面语义。",
            "再加入层级 planner、verifier、API/tool call、半在线修正或在线环境反馈。",
            "把长任务拆成子目标、状态检查和错误恢复，让每个模块都有可验证输出。",
            "最终在真实或半真实环境中检验整条 trajectory 的完成率，而不只看单步准确率。",
        ]
    return [
        "统一任务、观察、动作和奖励接口。",
        "收集或合成轨迹，过滤不可执行和不可验证样本。",
        "用 SFT/RFT/RL 更新策略。",
        "在目标 benchmark 或真实环境中按任务成功率评估。",
    ]


def benchmark_table_rows(paper: Paper) -> list[tuple[str, str, str, str]]:
    exp = paper.experiment
    benchmark = "论文/综述列出的 GUI benchmark"
    if "AndroidWorld" in exp:
        benchmark = "AndroidWorld / AndroidLab / AITW"
    elif "WebArena" in exp:
        benchmark = "WebArena / WebArena-Lite / WebShop"
    elif "ScreenSpot" in exp:
        benchmark = "ScreenSpot / ScreenSpot-Pro"
    elif "OSWorld" in exp:
        benchmark = "OSWorld / desktop CUA benchmark"
    elif "Mind2Web" in exp:
        benchmark = "Mind2Web / Multimodal-Mind2Web / MiniWoB++"
    elif "LIBERO" in exp:
        benchmark = "LIBERO / robotics benchmark"
    return [
        ("主 benchmark", benchmark, "任务成功率 / grounding accuracy / 系统吞吐", exp),
        ("对照基线", "SFT、BC、prompting、闭源 VLM 或同尺寸开源模型", "相同任务口径下比较", "不同论文的环境和任务集合不同，不能把数值直接跨表相加。"),
        ("可信度", "作者自评或综述转述", "⚠️ / 待核", confidence_note(paper)),
    ]


def limitations_for(paper: Paper) -> list[str]:
    tags = set(paper.tags)
    limits = []
    if "Grounding" in tags:
        limits.append("单步 grounding 分数高不等于多轮任务成功；坐标准但点错流程仍会失败。")
        limits.append("连续奖励或裁剪策略可能对特定 benchmark 有利，跨分辨率、跨应用仍需验证。")
    if "Online RL" in tags:
        limits.append("在线 rollout 成本高，浏览器、VM、ADB 和截图 I/O 往往限制可扩展性。")
        limits.append("自动 verifier 或 VLM judge 一旦误判，策略可能学习 reward hacking。")
    if "Offline RL" in tags:
        limits.append("离线轨迹覆盖不到的状态仍是风险；模型一旦偏离专家路径，可能缺少纠正样本。")
    if "World Model" in tags:
        limits.append("世界模型会产生 sim-to-real gap；虚拟页面里可行的动作不一定能在真实环境复现。")
    if "Data" in tags:
        limits.append("自动合成数据需要严格去重和可复现检查，否则会把伪成功、模板泄漏或不可执行指令带入训练。")
    if "OS/Desktop" in tags:
        limits.append("桌面任务环境重、状态复杂，评测结果容易受 VM 配置、软件版本和任务初始化影响。")
    if not limits:
        limits.append("论文结果大多是作者自报或综述转述，跨 benchmark 的数值不可直接混算。")
    return limits[:3]


def relation_for(paper: Paper) -> str:
    tags = set(paper.tags)
    if "World Model" in tags:
        return "在 GUI Agent RL 谱系中，它属于“绕过真实环境 I/O wall”的路线：用模型化的数字世界降低采样成本，再服务真实策略训练。"
    if "Grounding" in tags:
        return "它位于“把视觉定位做准”的支线：没有可靠 grounding，多轮策略再强也会在第一步点击或输入上失败。"
    if "Online RL" in tags:
        return "它属于闭环自我改进路线：让 agent 在环境里行动、失败、修正，而不是停留在离线示范模仿。"
    if "Offline RL" in tags:
        return "它属于安全低成本路线：先尽量从已有轨迹和可验证标签中榨取价值，再决定是否进入在线环境。"
    if "Hybrid" in tags:
        return "它属于系统融合路线：用层级结构、工具调用、验证器或半在线机制把真实 GUI 的长时序问题拆小。"
    return "它为 GUI Agent RL 提供数据、环境或系统支撑，是从单点算法走向可复现工程栈的一部分。"


def detail_page(paper: Paper, prev_paper: Paper | None, next_paper: Paper | None) -> str:
    notes = detail_notes_for(paper)
    task_rows = "\n".join(
        f"<tr><th>{inline_md(k)}</th><td>{inline_md(v)}</td></tr>" for k, v in task_form_for(paper)
    )
    profile_table = "\n".join(
        f"<tr><th>{inline_md(k)}</th><td>{inline_md(v)}</td></tr>" for k, v in profile_rows(paper)
    )
    pipeline_html = "\n".join(f"<li>{inline_md(step)}</li>" for step in pipeline_for(paper))
    notes_html = "\n".join(f"<li>{inline_md(note)}</li>" for note in notes)
    recipe_html = "\n".join(f"<li>{inline_md(item)}</li>" for item in training_recipe_for(paper))
    innovations_html = "\n".join(f"<li>{inline_md(item)}</li>" for item in innovations_for(paper))
    checklist_html = "\n".join(f"<li>{inline_md(item)}</li>" for item in implementation_checklist_for(paper))
    limits_html = "\n".join(f"<li>{inline_md(item)}</li>" for item in limitations_for(paper))
    benchmark_rows = "\n".join(
        f"<tr><td>{inline_md(a)}</td><td>{inline_md(b)}</td><td>{inline_md(c)}</td><td>{inline_md(d)}</td></tr>"
        for a, b, c, d in benchmark_table_rows(paper)
    )
    tags = "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in paper.tags)
    paper_links = "".join(
        f'<a class="primary-action" href="{html.escape(url)}" target="_blank" rel="noreferrer">打开论文</a>'
        for url in paper.links
    )
    if not paper_links and paper.link_text:
        paper_links = f'<span class="paper-note">{inline_md(paper.link_text)}</span>'
    prev_link = (
        f'<a href="{slug_for(prev_paper)}.html">上一篇：{inline_md(prev_paper.title)}</a>' if prev_paper else "<span></span>"
    )
    next_link = (
        f'<a href="{slug_for(next_paper)}.html">下一篇：{inline_md(next_paper.title)}</a>' if next_paper else "<span></span>"
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{inline_md(paper.title)} | GUI Agent RL 论文细读</title>
  <meta name="description" content="{html.escape(paper.title)} 论文细读：问题、方法、训练、奖励、实验与局限。">
  <link rel="icon" href="../favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="../styles.css">
</head>
<body>
  <header class="site-header">
    <nav class="nav-shell" aria-label="站点导航">
      <a class="brand" href="../index.html#top">GUI Agent RL</a>
      <div class="nav-links">
        <a href="../index.html#papers">论文列表</a>
        <a href="../index.html#env">环境</a>
        <a href="../index.html#refs">参考</a>
      </div>
    </nav>
  </header>
  <main class="paper-detail">
    <section class="detail-hero">
      <p class="eyebrow">Paper {paper.number:02d} · 论文细读</p>
      <h1>{inline_md(paper.title)}</h1>
      <div class="tag-row">{tags}</div>
      <p class="lead">{inline_md(paper.motivation)}</p>
      <div class="hero-actions">
        {paper_links}
        <a class="secondary-action" href="{SURVEY_URL}" target="_blank" rel="noreferrer">综述来源</a>
      </div>
    </section>

    <article class="detail-layout">
      <aside class="toc-panel" aria-label="本页目录">
        <a href="#tldr">TL;DR</a>
        <a href="#profile">定位卡</a>
        <a href="#problem">问题</a>
        <a href="#form">任务形式</a>
        <a href="#pipeline">闭环</a>
        <a href="#method">方法拆解</a>
        <a href="#training">训练与奖励</a>
        <a href="#innovation">创新</a>
        <a href="#experiment">实验</a>
        <a href="#reproduce">复现</a>
        <a href="#limits">局限</a>
      </aside>
      <div class="detail-content">
        <section id="tldr">
          <h2>TL;DR</h2>
          <p>{inline_md(paper.method)} {inline_md(paper.experiment)}</p>
          <blockquote><p>{inline_md(confidence_note(paper))}</p></blockquote>
        </section>
        <section id="profile">
          <h2>0. 论文定位卡</h2>
          <table class="detail-table"><tbody>{profile_table}</tbody></table>
          <p>先看这张卡可以避免误读：GUI Agent RL 论文经常把“模型能力”“环境系统”“奖励设计”“数据生成”混在一起讨论。这里把它拆成环境、动作空间、奖励信号和优化方式四个轴，方便判断论文到底贡献在哪里。</p>
        </section>
        <section id="problem">
          <h2>1. 要解决的问题</h2>
          <p>{inline_md(paper.motivation)}</p>
          <p>换成 GUI agent 的语言，这个问题通常落在三件事上：界面状态部分可观测、正确反馈稀疏且延迟、静态示范无法覆盖真实软件变化。论文的价值就在于把这些问题中的一个或多个转成可训练的闭环信号。</p>
          <p>如果只用 SFT 或行为克隆，模型学到的是“在数据集截图上下一步该怎么点”；而 GUI Agent 真正部署时会遇到状态漂移、异步加载、误点回退、任务参数变化和界面版本变化。本文所属路线试图回答的是：如何让模型从环境反馈、可验证标签或合成轨迹里继续改进，而不是停留在一次性模仿。</p>
        </section>
        <section id="form">
          <h2>2. 任务形式：输入、动作、奖励</h2>
          <table class="detail-table"><tbody>{task_rows}</tbody></table>
          <p>这个表是读 GUI Agent RL 论文最重要的入口：只要弄清楚模型看到了什么、能做什么、奖励从哪里来，就能判断它是算法贡献、环境贡献、奖励贡献还是系统贡献。</p>
        </section>
        <section id="pipeline">
          <h2>3. 训练闭环怎么跑</h2>
          <ol>{pipeline_html}</ol>
          <div class="flow-strip">
            <span>任务/轨迹</span><span>策略采样</span><span>环境或 verifier</span><span>奖励/筛选</span><span>策略更新</span>
          </div>
          <p>这条闭环是理解论文的主线。无论它叫 GRPO、DPO、AEPO、MCTS、world model 还是 semi-online，最终都要把“模型输出的动作”转换成“可执行状态变化”，再把状态变化转换成“可训练信号”。</p>
        </section>
        <section id="method">
          <h2>4. 方法与架构怎么跑</h2>
          <ol>{notes_html}</ol>
          <p>因此，这篇论文不是简单地“让大模型点屏幕”，而是围绕 observation-action-reward 契约重写训练闭环：先让动作可执行，再让反馈可验证，最后让策略能从成功和失败中更新。</p>
          <h3>4.1 关键中间变量</h3>
          <p>读方法时要特别盯住中间变量：轨迹是按 step 存、按 episode 存，还是按候选动作组存？奖励是只在终局出现，还是每一步都有 progress？模型更新时用的是整条轨迹的相对优势、单步坐标误差、偏好对，还是 value/Q head？这些细节决定了论文能否处理长任务和稀疏奖励。</p>
          <h3>4.2 为什么这种设计能解决原问题</h3>
          <p>核心逻辑是把原本不可微、不可直接监督的 GUI 成功条件拆成更接近训练信号的形式：要么用规则/verifier 把成功自动判出来，要么用 reward model 学过程进展，要么用世界模型/搜索生成更多成功轨迹，要么用层级结构把几十步任务拆成几个短子目标。</p>
        </section>
        <section id="training">
          <h2>5. 训练、奖励与数据流</h2>
          <ul>{recipe_html}</ul>
          <p>读实验时要特别注意 reward 口径：有的论文评估单步坐标，有的评估整条任务成功，有的只报告系统吞吐或数据合成质量。不同口径不能直接相加或横向混算。</p>
        </section>
        <section id="innovation">
          <h2>6. 关键设计与创新点</h2>
          <ul>{innovations_html}</ul>
          <p>这些创新点的共同目标，是把 GUI 交互从一次性 prompt 调用变成可迭代优化的训练系统。</p>
        </section>
        <section id="experiment">
          <h2>7. 实验怎么读</h2>
          <p>{inline_md(paper.experiment)}</p>
          <table class="result-table">
            <thead><tr><th>项目</th><th>口径</th><th>指标</th><th>结论/注意事项</th></tr></thead>
            <tbody>{benchmark_rows}</tbody>
          </table>
          <p>这类实验通常需要同时看三组数字：第一是任务成功率或 grounding accuracy；第二是与 SFT、BC、prompting、GPT-4V/闭源模型或开源基线的对比；第三是环境成本，例如 rollout 速度、样本数量、标注成本和并行规模。</p>
          <p>尤其要避免一个常见误读：ScreenSpot/grounding accuracy、WebArena success rate、AndroidWorld task success、OSWorld desktop success、world-model rollout speed 不是同一个指标。本文页面保留原论文或综述的原始口径，不把它们强行换算。</p>
        </section>
        <section id="reproduce">
          <h2>8. 复现或落地时要准备什么</h2>
          <ol>{checklist_html}</ol>
          <p>如果只能先复现一小部分，建议优先复现 action schema、verifier 和一组可重置任务；没有这三件事，后续 RL 指标很难可信。</p>
        </section>
        <section id="limits">
          <h2>9. 局限与读法</h2>
          <ul>{limits_html}</ul>
          <p>{inline_md(relation_for(paper))}</p>
          <p>和参考站的可信度规则一致，本页把作者自评、技术报告自报和综述转述都按 ⚠️ 处理；只有基准维护方统一评测或第三方复现才适合当作更强证据。</p>
        </section>
        <section id="sources">
          <h2>来源</h2>
          <p>本页依据 GUI Agent RL 综述、论文摘要/公开页面和站内总报告整理；数值优先保留论文自报口径，未做第三方复现实验。参考站体例来自具身智能学习站的论文细读页面，例如 RT-2 与 SimpleVLA-RL 细读。</p>
          <div class="source-links">{paper_links}<a href="../index.html#refs">返回参考链接表</a></div>
        </section>
      </div>
    </article>
    <nav class="pager" aria-label="论文翻页">{prev_link}{next_link}</nav>
  </main>
  <footer class="site-footer">
    <p><a href="../index.html#papers">返回论文列表</a></p>
  </footer>
</body>
</html>
"""


def paper_card(paper: Paper) -> str:
    link_buttons = "".join(
        f'<a class="paper-link" href="{html.escape(url)}" target="_blank" rel="noreferrer">论文链接</a>'
        for url in paper.links
    )
    if not link_buttons and paper.link_text:
        link_buttons = f'<span class="paper-note">{inline_md(paper.link_text)}</span>'
    tags = "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in paper.tags)
    searchable = html.escape((paper.title + " " + " ".join(paper.tags)).lower())
    primary = html.escape(paper.tags[0])
    slug = slug_for(paper)
    return f"""
    <article class="paper-card" id="paper-{paper.number}" data-tags="{' '.join(html.escape(t) for t in paper.tags)}" data-primary="{primary}" data-search="{searchable}">
      <div class="paper-top">
        <a class="paper-no" href="papers/{slug}.html">{paper.number:02d}</a>
        <div>
          <h3><a href="papers/{slug}.html">{inline_md(paper.title)}</a></h3>
          <div class="tag-row">{tags}</div>
        </div>
        <div class="paper-actions">{link_buttons}</div>
      </div>
      <div class="paper-grid">
        <section><h4>动机</h4><p>{inline_md(paper.motivation)}</p></section>
        <section><h4>方案</h4><p>{inline_md(paper.method)}</p></section>
        <section><h4>实验</h4><p>{inline_md(paper.experiment)}</p></section>
      </div>
      <div class="card-readmore"><a href="papers/{slug}.html">进入完整细读</a></div>
    </article>
    """


def build() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    overview = extract_between(source, "## 0. 总体判断", "## 1. 核心方法论文")
    paper_section = extract_between(source, "## 1. 核心方法论文", "## 2. 训练环境、容器与技术栈")
    env_section = extract_between(source, "## 2. 训练环境、容器与技术栈", "## 3. 现在大家实际怎么做 GUI Agent RL")
    practice = extract_between(source, "## 3. 现在大家实际怎么做 GUI Agent RL", "## 4. 关键趋势")
    trends = extract_between(source, "## 4. 关键趋势", "## 5. 参考链接")
    refs = parse_references(extract_between(source, "## 5. 参考链接"))
    papers = parse_papers(paper_section)
    envs = parse_heading_sections(env_section)
    paper_dir = ROOT / "papers"
    paper_dir.mkdir(exist_ok=True)
    for old_page in paper_dir.glob("*.html"):
        old_page.unlink()

    papers_html = "\n".join(paper_card(p) for p in papers)
    env_html = "\n".join(
        f'<article class="env-panel" id="env-{idx + 1}"><h3>{inline_md(title)}</h3>{body}</article>'
        for idx, (title, body) in enumerate(envs)
    )
    refs_html = "\n".join(
        f'<a href="{html.escape(url)}" target="_blank" rel="noreferrer">{html.escape(name)}</a>' for name, url in refs
    )

    overview_lines = [line.strip() for line in overview.splitlines() if line.strip()]
    thesis = overview_lines[0] if overview_lines else ""
    route_lines = [line for line in overview_lines if re.match(r"\d+\.", line)]
    routes_html = "\n".join(
        f'<div class="route"><strong>{inline_md(line.split("：", 1)[0])}</strong><span>{inline_md(line.split("：", 1)[1] if "：" in line else line)}</span></div>'
        for line in route_lines[:3]
    )

    practice_html = block_md(practice.splitlines())
    trends_html = block_md(trends.splitlines())

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GUI Agent RL 论文地图</title>
  <meta name="description" content="GUI Agent with Reinforcement Learning 论文梳理、环境技术栈与训练路线。">
  <link rel="icon" href="favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <a class="skip-link" href="#papers">跳到论文列表</a>
  <header class="site-header">
    <nav class="nav-shell" aria-label="站点导航">
      <a class="brand" href="#top">GUI Agent RL</a>
      <div class="nav-links">
        <a href="#map">路线</a>
        <a href="#papers">论文</a>
        <a href="#env">环境</a>
        <a href="#practice">做法</a>
        <a href="#refs">参考</a>
      </div>
    </nav>
  </header>

  <main id="top">
    <section class="hero" aria-labelledby="hero-title">
      <div class="hero-copy">
        <p class="eyebrow">arXiv:2604.27955 阅读笔记</p>
        <h1 id="hero-title">GUI Agent RL 论文地图</h1>
        <p class="lead">{inline_md(thesis)}</p>
        <div class="hero-actions">
          <a class="primary-action" href="#papers">浏览 41 篇论文</a>
          <a class="secondary-action" href="https://arxiv.org/abs/2604.27955" target="_blank" rel="noreferrer">打开综述原文</a>
        </div>
      </div>
      <aside class="signal-panel" aria-label="报告范围">
        <dl>
          <div><dt>论文</dt><dd>{len(papers)}</dd></div>
          <div><dt>路线</dt><dd>3</dd></div>
          <div><dt>环境</dt><dd>Web / OS / Mobile</dd></div>
        </dl>
      </aside>
    </section>

    <section class="section-block" id="map" aria-labelledby="map-title">
      <div class="section-heading">
        <p class="eyebrow">Taxonomy</p>
        <h2 id="map-title">三条主线</h2>
      </div>
      <div class="route-grid">{routes_html}</div>
    </section>

    <section class="section-block" id="papers" aria-labelledby="papers-title">
      <div class="section-heading wide">
        <div>
          <p class="eyebrow">Paper Notes</p>
          <h2 id="papers-title">逐篇论文细读</h2>
        </div>
        <div class="toolbar" role="search">
          <input id="paper-search" type="search" placeholder="搜索论文、路线或环境" aria-label="搜索论文">
          <select id="paper-filter" aria-label="按类别筛选">
            <option value="all">全部类别</option>
            <option>Online RL</option>
            <option>Offline RL</option>
            <option>Grounding</option>
            <option>Hybrid</option>
            <option>World Model</option>
            <option>Reward</option>
            <option>Web</option>
            <option>Mobile</option>
            <option>OS/Desktop</option>
          </select>
        </div>
      </div>
      <p class="result-line"><span id="paper-count">{len(papers)}</span> 篇论文显示中</p>
      <div class="paper-list">{papers_html}</div>
    </section>

    <section class="section-block" id="env" aria-labelledby="env-title">
      <div class="section-heading">
        <p class="eyebrow">Environments</p>
        <h2 id="env-title">环境、容器与框架</h2>
      </div>
      <div class="env-grid">{env_html}</div>
    </section>

    <section class="section-block split" id="practice" aria-labelledby="practice-title">
      <div class="section-heading sticky-heading">
        <p class="eyebrow">Recipe</p>
        <h2 id="practice-title">现在大家实际怎么做</h2>
      </div>
      <div class="prose">{practice_html}</div>
    </section>

    <section class="section-block split" id="trends" aria-labelledby="trends-title">
      <div class="section-heading sticky-heading">
        <p class="eyebrow">Trends</p>
        <h2 id="trends-title">关键趋势</h2>
      </div>
      <div class="prose">{trends_html}</div>
    </section>

    <section class="section-block" id="refs" aria-labelledby="refs-title">
      <div class="section-heading">
        <p class="eyebrow">Sources</p>
        <h2 id="refs-title">参考链接</h2>
      </div>
      <div class="refs-grid">{refs_html}</div>
    </section>
  </main>

  <footer class="site-footer">
    <p>静态站点，由 Markdown 报告自动生成，可直接部署到 GitHub Pages。</p>
  </footer>

  <script src="script.js"></script>
</body>
</html>
"""

    (ROOT / "index.html").write_text(html_doc, encoding="utf-8")
    (ROOT / "styles.css").write_text(CSS, encoding="utf-8")
    (ROOT / "script.js").write_text(JS, encoding="utf-8")
    (ROOT / "favicon.svg").write_text(FAVICON, encoding="utf-8")
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    (ROOT / "README.md").write_text(README, encoding="utf-8")
    for idx, paper in enumerate(papers):
        prev_paper = papers[idx - 1] if idx > 0 else None
        next_paper = papers[idx + 1] if idx + 1 < len(papers) else None
        (paper_dir / f"{slug_for(paper)}.html").write_text(detail_page(paper, prev_paper, next_paper), encoding="utf-8")


CSS = r"""
:root {
  color-scheme: light;
  --paper: oklch(96.8% 0.018 78);
  --paper-strong: oklch(91.5% 0.024 78);
  --ink: oklch(21% 0.027 255);
  --muted: oklch(43% 0.035 252);
  --rule: oklch(77% 0.035 75);
  --red: oklch(55% 0.18 27);
  --blue: oklch(41% 0.12 236);
  --green: oklch(47% 0.10 155);
  --amber: oklch(70% 0.14 82);
  --shadow: 0 20px 60px color-mix(in oklch, var(--ink) 15%, transparent);
  --radius: 8px;
  --max: 1180px;
  --sans: ui-sans-serif, "Avenir Next", "PingFang SC", "Microsoft YaHei", sans-serif;
  --serif: Georgia, "Songti SC", "Noto Serif CJK SC", serif;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  background:
    linear-gradient(90deg, color-mix(in oklch, var(--rule) 18%, transparent) 1px, transparent 1px) 0 0 / 44px 44px,
    linear-gradient(0deg, color-mix(in oklch, var(--rule) 16%, transparent) 1px, transparent 1px) 0 0 / 44px 44px,
    var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  line-height: 1.72;
  letter-spacing: 0;
}

a {
  color: var(--blue);
  text-decoration-thickness: 0.08em;
  text-underline-offset: 0.18em;
}

.skip-link {
  position: absolute;
  left: 1rem;
  top: -4rem;
  z-index: 100;
  background: var(--ink);
  color: var(--paper);
  padding: 0.55rem 0.8rem;
  border-radius: var(--radius);
}

.skip-link:focus {
  top: 1rem;
}

.site-header {
  position: sticky;
  top: 0;
  z-index: 20;
  border-bottom: 1px solid color-mix(in oklch, var(--rule) 70%, transparent);
  background: color-mix(in oklch, var(--paper) 88%, transparent);
  backdrop-filter: blur(14px);
}

.nav-shell {
  width: min(var(--max), calc(100% - 32px));
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 64px;
  gap: 1rem;
}

.brand {
  color: var(--ink);
  font-weight: 800;
  text-decoration: none;
  letter-spacing: 0;
}

.nav-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  justify-content: flex-end;
}

.nav-links a {
  color: var(--muted);
  text-decoration: none;
  padding: 0.35rem 0.62rem;
  border-radius: var(--radius);
  font-size: 0.92rem;
}

.nav-links a:hover,
.nav-links a:focus-visible {
  background: var(--paper-strong);
  color: var(--ink);
}

main {
  width: min(var(--max), calc(100% - 32px));
  margin: 0 auto;
}

.hero {
  min-height: calc(100svh - 64px);
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(260px, 360px);
  align-items: center;
  gap: clamp(2rem, 6vw, 5rem);
  padding: clamp(3rem, 8vw, 7rem) 0 clamp(2rem, 5vw, 4rem);
  border-bottom: 2px solid var(--ink);
}

.eyebrow {
  margin: 0 0 0.65rem;
  color: var(--red);
  font-size: 0.78rem;
  font-weight: 800;
  text-transform: uppercase;
}

h1,
h2,
h3,
h4 {
  margin: 0;
  line-height: 1.15;
  letter-spacing: 0;
}

h1 {
  max-width: 850px;
  font-family: var(--serif);
  font-size: clamp(3.2rem, 8vw, 7.4rem);
  font-weight: 700;
}

.lead {
  max-width: 760px;
  margin: 1.5rem 0 0;
  color: color-mix(in oklch, var(--ink) 82%, var(--muted));
  font-size: clamp(1rem, 1.8vw, 1.22rem);
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 2rem;
}

.primary-action,
.secondary-action,
.paper-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 40px;
  padding: 0.55rem 0.9rem;
  border: 1px solid var(--ink);
  border-radius: var(--radius);
  text-decoration: none;
  font-weight: 750;
}

.primary-action {
  background: var(--ink);
  color: var(--paper);
}

.secondary-action,
.paper-link {
  color: var(--ink);
  background: color-mix(in oklch, var(--paper) 84%, white);
}

.signal-panel {
  align-self: stretch;
  display: grid;
  align-content: end;
}

.signal-panel dl {
  display: grid;
  gap: 1px;
  margin: 0;
  border: 2px solid var(--ink);
  background: var(--ink);
  box-shadow: var(--shadow);
}

.signal-panel div {
  padding: 1.1rem;
  background: var(--paper-strong);
}

.signal-panel dt {
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 750;
}

.signal-panel dd {
  margin: 0.18rem 0 0;
  font-family: var(--serif);
  font-size: clamp(1.4rem, 3vw, 2.4rem);
  font-weight: 700;
}

.section-block {
  padding: clamp(3rem, 7vw, 5.5rem) 0;
  border-bottom: 1px solid color-mix(in oklch, var(--rule) 80%, transparent);
}

.section-heading {
  max-width: 760px;
  margin-bottom: clamp(1.5rem, 4vw, 2.7rem);
}

.section-heading.wide {
  max-width: none;
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 1rem;
}

h2 {
  font-family: var(--serif);
  font-size: clamp(2.1rem, 5vw, 4rem);
}

.route-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
}

.route {
  min-height: 220px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 1.5rem;
  padding: 1.1rem;
  border-top: 4px solid var(--ink);
  background: color-mix(in oklch, var(--paper) 92%, white);
}

.route strong {
  font-size: clamp(1.1rem, 2vw, 1.45rem);
}

.route span {
  color: var(--muted);
}

.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  justify-content: flex-end;
}

input,
select {
  min-height: 42px;
  border: 1px solid var(--ink);
  border-radius: var(--radius);
  background: color-mix(in oklch, var(--paper) 88%, white);
  color: var(--ink);
  font: inherit;
  padding: 0.45rem 0.7rem;
}

input {
  width: min(320px, 100%);
}

.result-line {
  margin: -1.2rem 0 1.1rem;
  color: var(--muted);
  font-size: 0.94rem;
}

.paper-list {
  display: grid;
  gap: 1rem;
}

.paper-card {
  background: color-mix(in oklch, var(--paper) 90%, white);
  border: 1px solid color-mix(in oklch, var(--ink) 24%, var(--rule));
  border-radius: var(--radius);
  overflow: hidden;
}

.paper-card[hidden] {
  display: none;
}

.paper-top {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 1rem;
  padding: 1rem;
  border-bottom: 1px solid color-mix(in oklch, var(--rule) 75%, transparent);
}

.paper-no {
  color: var(--red);
  font-family: var(--serif);
  font-size: 1.5rem;
  font-weight: 700;
  text-decoration: none;
}

.paper-top h3 {
  font-size: clamp(1.05rem, 2vw, 1.35rem);
}

.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.55rem;
}

.tag {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 0.08rem 0.45rem;
  border: 1px solid color-mix(in oklch, var(--ink) 35%, var(--rule));
  border-radius: 999px;
  color: color-mix(in oklch, var(--muted) 80%, var(--ink));
  font-size: 0.74rem;
  font-weight: 700;
}

.paper-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  justify-content: flex-end;
}

.paper-link {
  min-height: 32px;
  padding: 0.25rem 0.55rem;
  font-size: 0.8rem;
}

.paper-note {
  max-width: 240px;
  color: var(--muted);
  font-size: 0.86rem;
}

.paper-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.paper-grid section {
  padding: 1rem;
  border-right: 1px solid color-mix(in oklch, var(--rule) 70%, transparent);
}

.paper-grid section:last-child {
  border-right: 0;
}

.paper-grid h4 {
  color: var(--red);
  font-size: 0.82rem;
  margin-bottom: 0.45rem;
}

.paper-grid p,
.prose p,
.env-panel p,
.env-panel li,
.prose li {
  margin: 0;
  color: color-mix(in oklch, var(--ink) 82%, var(--muted));
  font-size: 0.96rem;
}

.env-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

.env-panel {
  padding: 1rem;
  border-left: 4px solid var(--green);
  background: color-mix(in oklch, var(--paper) 90%, white);
}

.env-panel h3 {
  margin-bottom: 0.75rem;
  font-size: 1.25rem;
}

.env-panel p + p,
.prose p + p,
.prose ul + p,
.env-panel ul + p {
  margin-top: 0.8rem;
}

.env-panel ul,
.prose ul {
  margin: 0.55rem 0 0;
  padding-left: 1.15rem;
}

.split {
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr);
  gap: clamp(1.5rem, 5vw, 4rem);
}

.sticky-heading {
  position: sticky;
  top: 88px;
  align-self: start;
}

.prose {
  max-width: 760px;
}

.refs-grid {
  columns: 3 220px;
  column-gap: 1rem;
}

.refs-grid a {
  display: block;
  break-inside: avoid;
  margin: 0 0 0.42rem;
  color: var(--blue);
}

.card-readmore {
  padding: 0.75rem 1rem 1rem;
  border-top: 1px solid color-mix(in oklch, var(--rule) 70%, transparent);
}

.card-readmore a,
.paper-top h3 a {
  color: var(--ink);
  text-decoration: none;
}

.card-readmore a {
  color: var(--blue);
  font-weight: 800;
}

.paper-detail {
  width: min(var(--max), calc(100% - 32px));
}

.detail-hero {
  padding: clamp(3rem, 7vw, 5.5rem) 0 clamp(2rem, 5vw, 4rem);
  border-bottom: 2px solid var(--ink);
}

.detail-hero h1 {
  max-width: 980px;
  font-size: clamp(2.4rem, 6vw, 5.2rem);
}

.detail-layout {
  display: grid;
  grid-template-columns: 230px minmax(0, 1fr);
  gap: clamp(1.5rem, 5vw, 4rem);
  padding: clamp(2.5rem, 6vw, 4rem) 0;
}

.toc-panel {
  position: sticky;
  top: 88px;
  align-self: start;
  display: grid;
  gap: 0.25rem;
  padding: 0.85rem;
  border: 1px solid color-mix(in oklch, var(--ink) 25%, var(--rule));
  background: color-mix(in oklch, var(--paper) 90%, white);
}

.toc-panel a {
  color: var(--muted);
  text-decoration: none;
  padding: 0.28rem 0.35rem;
  border-radius: 4px;
  font-size: 0.92rem;
  font-weight: 720;
}

.toc-panel a:hover,
.toc-panel a:focus-visible {
  background: var(--paper-strong);
  color: var(--ink);
}

.detail-content {
  max-width: 840px;
}

.detail-content section {
  padding: 0 0 clamp(2rem, 5vw, 3.5rem);
  margin-bottom: clamp(2rem, 5vw, 3.5rem);
  border-bottom: 1px solid color-mix(in oklch, var(--rule) 78%, transparent);
  scroll-margin-top: 96px;
}

.detail-content h2 {
  font-size: clamp(1.7rem, 3vw, 2.6rem);
  margin-bottom: 1rem;
}

.detail-content h3 {
  margin: 1.25rem 0 0.55rem;
  font-size: clamp(1.12rem, 2vw, 1.38rem);
}

.detail-content p,
.detail-content li,
.detail-table td {
  color: color-mix(in oklch, var(--ink) 84%, var(--muted));
  font-size: 1rem;
}

.detail-content blockquote {
  margin: 1rem 0 0;
  padding: 0.8rem 1rem;
  border-left: 4px solid var(--red);
  background: color-mix(in oklch, var(--amber) 18%, transparent);
}

.detail-content blockquote p {
  margin: 0;
  color: var(--ink);
  font-weight: 720;
}

.detail-content p + p,
.detail-content ul + p,
.detail-content ol + p {
  margin-top: 0.9rem;
}

.detail-content ol,
.detail-content ul {
  padding-left: 1.25rem;
  margin: 0;
}

.detail-content li + li {
  margin-top: 0.65rem;
}

.detail-table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 1rem;
  background: color-mix(in oklch, var(--paper) 92%, white);
}

.detail-table th,
.detail-table td {
  border: 1px solid color-mix(in oklch, var(--rule) 80%, var(--ink));
  padding: 0.8rem;
  vertical-align: top;
  text-align: left;
}

.detail-table th {
  width: 7rem;
  color: var(--red);
  font-size: 0.9rem;
}

.result-table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  background: color-mix(in oklch, var(--paper) 90%, white);
}

.result-table th,
.result-table td {
  border: 1px solid color-mix(in oklch, var(--rule) 80%, var(--ink));
  padding: 0.72rem;
  vertical-align: top;
  text-align: left;
  font-size: 0.93rem;
}

.result-table th {
  color: var(--red);
  background: var(--paper-strong);
}

.flow-strip {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.45rem;
  margin: 1rem 0;
}

.flow-strip span {
  min-height: 52px;
  display: grid;
  place-items: center;
  padding: 0.45rem;
  border: 1px solid var(--ink);
  background: color-mix(in oklch, var(--paper) 88%, white);
  color: var(--ink);
  text-align: center;
  font-size: 0.82rem;
  font-weight: 800;
}

.source-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
}

.source-links a {
  min-height: 36px;
  display: inline-flex;
  align-items: center;
  padding: 0.35rem 0.65rem;
  border: 1px solid color-mix(in oklch, var(--ink) 40%, var(--rule));
  border-radius: var(--radius);
  text-decoration: none;
  font-weight: 750;
}

.pager {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: 1.25rem 0 3rem;
  border-top: 2px solid var(--ink);
}

.pager a {
  max-width: 48%;
  color: var(--ink);
  font-weight: 800;
  text-decoration: none;
}

.site-footer {
  width: min(var(--max), calc(100% - 32px));
  margin: 0 auto;
  padding: 2rem 0 3rem;
  color: var(--muted);
  font-size: 0.92rem;
}

code {
  background: color-mix(in oklch, var(--amber) 28%, transparent);
  padding: 0.05rem 0.24rem;
  border-radius: 4px;
}

@media (max-width: 860px) {
  .hero,
  .route-grid,
  .paper-grid,
  .env-grid,
  .split,
  .detail-layout {
    grid-template-columns: 1fr;
  }

  .hero {
    min-height: auto;
  }

  .section-heading.wide,
  .paper-top {
    display: block;
  }

  .toolbar,
  .paper-actions {
    justify-content: flex-start;
    margin-top: 0.85rem;
  }

  input {
    width: 100%;
  }

  .paper-grid section {
    border-right: 0;
    border-bottom: 1px solid color-mix(in oklch, var(--rule) 70%, transparent);
  }

  .paper-grid section:last-child {
    border-bottom: 0;
  }

  .sticky-heading {
    position: static;
  }

  .toc-panel {
    position: static;
  }

  .pager {
    display: grid;
  }

  .pager a {
    max-width: none;
  }

  .flow-strip {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 560px) {
  .nav-shell {
    align-items: flex-start;
    flex-direction: column;
    padding: 0.75rem 0;
  }

  .nav-links {
    justify-content: flex-start;
  }

  h1 {
    font-size: clamp(2.7rem, 18vw, 4.1rem);
  }
}
"""

JS = r"""
const search = document.querySelector("#paper-search");
const filter = document.querySelector("#paper-filter");
const cards = [...document.querySelectorAll(".paper-card")];
const count = document.querySelector("#paper-count");

function applyFilters() {
  const query = (search.value || "").trim().toLowerCase();
  const selected = filter.value;
  let visible = 0;

  for (const card of cards) {
    const text = card.dataset.search || "";
    const tags = card.dataset.tags || "";
    const matchesQuery = !query || text.includes(query);
    const matchesTag = selected === "all" || tags.includes(selected);
    const show = matchesQuery && matchesTag;
    card.hidden = !show;
    if (show) visible += 1;
  }

  count.textContent = String(visible);
}

search?.addEventListener("input", applyFilters);
filter?.addEventListener("change", applyFilters);
applyFilters();

document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", () => {
    const target = document.querySelector(anchor.getAttribute("href"));
    if (!target) return;
    target.setAttribute("tabindex", "-1");
    setTimeout(() => target.focus({ preventScroll: true }), 350);
  });
});
"""

FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="8" fill="#f3ecdf"/>
  <path d="M12 14h40v34H12z" fill="none" stroke="#111827" stroke-width="4"/>
  <path d="M18 24h28M18 33h20M18 42h14" stroke="#b33b2e" stroke-width="4" stroke-linecap="square"/>
</svg>
"""

README = """# RL4GUIAgent

这是一个可直接部署到 GitHub Pages 的 GUI Agent RL 论文细读站点。

站点结构：

- `index.html`：总览、路线、搜索、筛选、环境/容器/框架总结。
- `papers/*.html`：41 篇论文独立细读页，包含 TL;DR、问题、任务形式、方法拆解、训练/奖励、实验、复现清单、局限。
- `build_site.py`：从上一级目录的 `gui_agent_rl_survey.md` 生成静态站点。

## 本地预览

```bash
python3 -m http.server 8000
```

然后打开 <http://localhost:8000>。

## 重新生成

```bash
python3 build_site.py
```

## 发布到 GitHub Pages

仓库地址：

```bash
git@github.com:xrose3159/RL4GUIAgent.git
```

推送后在 GitHub 仓库设置中开启 Pages：

- Source: Deploy from a branch
- Branch: `main`
- Folder: `/root`

发布成功后，站点地址通常为：

```text
https://xrose3159.github.io/RL4GUIAgent/
```
"""


if __name__ == "__main__":
    build()
