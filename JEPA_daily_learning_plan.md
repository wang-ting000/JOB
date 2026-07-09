# JEPA 逐日学习计划

起始日期: 2026-07-09  
结束日期: 2026-09-02  
周期: 8 周，56 天  
主线目标: 从 I-JEPA 概念理解，推进到 V-JEPA 视频预测，再推进到 action-conditioned latent world model 和 planning 小项目。  
默认强度: 工作日 1.5 到 2.5 小时，周末 3 到 5 小时。  

## 最终交付物

1. 一份 JEPA 机制笔记: 能讲清楚 I-JEPA、V-JEPA、V-JEPA 2 的区别。
2. 一个 mini-I-JEPA 实验: 跑通图像 latent prediction、自监督预训练、linear probe。
3. 一个 mini-V-JEPA 实验: 跑通视频 latent prediction 或 toy video 表征学习。
4. 一个 latent world model planning demo: 图像观测编码成 latent，action-conditioned predictor 做 rollout，CEM/MPC 搜索动作。
5. 一页课题和实习表达材料: 能把 JEPA 和 world model + planning 讲成一个研究方向。

## 核心资料

- I-JEPA paper: https://arxiv.org/abs/2301.08243
- I-JEPA code: https://github.com/facebookresearch/ijepa
- V-JEPA paper: https://arxiv.org/abs/2404.08471
- V-JEPA code: https://github.com/facebookresearch/jepa
- V-JEPA 2 paper: https://arxiv.org/abs/2506.09985
- V-JEPA 2 code: https://github.com/facebookresearch/vjepa2
- EB-JEPA examples: https://github.com/facebookresearch/eb_jepa

## 每日固定流程

每天结束前写 5 行日志:

```text
今天学到的核心概念:
我能举出的例子:
我还不清楚的点:
今天的产出文件或结果:
明天第一步:
```

每周结束前做一次复盘:

```text
本周最重要的概念:
本周最容易混淆的概念:
本周代码或实验是否跑通:
下一周需要降低难度还是加速:
```

## 第 1 周: 建立 JEPA 直觉

目标: 明确 JEPA 为什么预测 latent，而不是预测 pixel。能把 I-JEPA 和 MAE、DINO、BYOL 区分开。

| Day | 日期 | 主题 | 今日任务 | 今日产出 |
|---|---|---|---|---|
| 1 | 2026-07-09 周四 | JEPA 总体直觉 | 阅读 I-JEPA 摘要、方法图、Introduction。只抓一个问题: 它预测的到底是什么。 | `notes/day01_what_is_jepa.md`: 5 句话解释 JEPA |
| 2 | 2026-07-10 周五 | 自监督学习版图 | 对比 contrastive learning、masked reconstruction、joint embedding。 | 一张三列表: SimCLR/MAE/I-JEPA 的输入、目标、loss |
| 3 | 2026-07-11 周六 | MAE vs I-JEPA | 阅读 MAE 思路概要，再回看 I-JEPA 的非生成式目标。 | 画出 MAE 和 I-JEPA 两张流程图 |
| 4 | 2026-07-12 周日 | 表征坍塌 | 学 BYOL/DINO 中 online encoder、target encoder、stop-gradient 的作用。 | 写出“为什么只预测 latent 不一定会坍塌”的解释 |
| 5 | 2026-07-13 周一 | ViT 基础 | 复习 patch embedding、positional embedding、CLS token、Transformer block。 | 手写 ViT 数据流: image -> patches -> tokens -> features |
| 6 | 2026-07-14 周二 | I-JEPA 架构拆解 | 拆 I-JEPA 的 context encoder、target encoder、predictor、mask。 | 一张 I-JEPA 模块表: 每个模块的输入输出 |
| 7 | 2026-07-15 周三 | 第 1 周复盘 | 用自己的话回答: JEPA 为什么可能更适合 world model。 | `week01_review.md`: 直觉、误区、迁移例子 |

理解检查:

- 如果 JEPA 不重建像素，它怎么知道自己预测对了？
- 为什么预测 latent 可能比预测 pixel 更适合学习语义？
- 如果把输入换成轨迹、无线信道或机器人视频，context 和 target 分别是什么？

## 第 2 周: 吃透 I-JEPA 训练机制

目标: 能读懂 I-JEPA 方法部分，并准备实现 mini-I-JEPA。

| Day | 日期 | 主题 | 今日任务 | 今日产出 |
|---|---|---|---|---|
| 8 | 2026-07-16 周四 | Mask strategy | 精读 I-JEPA 中 target block scale、context block 的设计动机。 | 写出 mask scale 如何影响语义层次 |
| 9 | 2026-07-17 周五 | Target encoder | 理解 EMA target encoder，写出参数更新公式。 | `ema_encoder_note.md`: EMA 直觉和公式 |
| 10 | 2026-07-18 周六 | Predictor | 理解 predictor 为什么需要 target position embedding。 | 画出 `z_context + target_pos -> z_pred` |
| 11 | 2026-07-19 周日 | Loss function | 比较 latent MSE、SmoothL1、cosine loss 的直觉差异。 | 写出 mini-I-JEPA 采用哪种 loss 以及理由 |
| 12 | 2026-07-20 周一 | Evaluation | 学 linear probe、fine-tune、kNN eval 的差别。 | 写出你要先做 linear probe 的原因 |
| 13 | 2026-07-21 周二 | 代码阅读 | 浏览 I-JEPA 官方仓库结构，找 masking、model、training loop。 | `ijepa_code_map.md`: 文件结构和关键函数猜测 |
| 14 | 2026-07-22 周三 | 第 2 周复盘 | 不看论文，默写 I-JEPA 训练流程。 | `week02_review.md`: 伪代码版 I-JEPA |

理解检查:

- target encoder 为什么不用梯度直接更新？
- target block 太小会发生什么？
- context block 太稀疏会发生什么？

## 第 3 周: 实现 mini-I-JEPA

目标: 用小数据集跑通图像 JEPA 原型。不追求 SOTA，只追求闭环。

| Day | 日期 | 主题 | 今日任务 | 今日产出 |
|---|---|---|---|---|
| 15 | 2026-07-23 周四 | 项目初始化 | 建立 `mini_jepa/` 目录，确定数据集。推荐 CIFAR-10、STL-10 或 Tiny-ImageNet。 | 项目目录和 README 草稿 |
| 16 | 2026-07-24 周五 | Data pipeline | 写 dataset、transform、dataloader。先跑通一个 batch。 | 打印 batch shape，保存一张可视化样本 |
| 17 | 2026-07-25 周六 | Patchify | 实现 patch embedding 或直接使用简化 ViT。 | 单元测试: 输入图像能变成 token 序列 |
| 18 | 2026-07-26 周日 | Masking | 实现 context mask 和 target mask。 | 可视化 mask 后的图像 patch 位置 |
| 19 | 2026-07-27 周一 | Encoder | 实现 online encoder 和 target encoder。 | 前向传播跑通: 输出 `z_context` 和 `z_target` |
| 20 | 2026-07-28 周二 | Predictor | 实现 predictor: 用 context latent 和 target position 预测 target latent。 | 前向传播跑通: 输出 `z_pred` |
| 21 | 2026-07-29 周三 | 第 3 周复盘 | 整合训练 step，但不要求长时间训练。 | `train_step.py` 能完成一次 loss backward |

理解检查:

- 你的 mask 是按 patch 随机采样，还是按 block 采样？
- `z_pred` 和 `z_target` 的 shape 是否严格一致？
- predictor 看到 target patch 的像素了吗？如果看到了，实验就泄漏了。

## 第 4 周: 训练、评估和消融 mini-I-JEPA

目标: 得到第一条 loss 曲线、一个 linear probe 结果、一个消融结论。

| Day | 日期 | 主题 | 今日任务 | 今日产出 |
|---|---|---|---|---|
| 22 | 2026-07-30 周四 | Training loop | 写完整训练循环、保存 checkpoint、记录 loss。 | 第一条 loss 曲线 |
| 23 | 2026-07-31 周五 | 稳定性检查 | 检查 loss 是否下降，排查 NaN、shape、EMA 更新。 | `debug_log.md`: 记录 3 个检查点 |
| 24 | 2026-08-01 周六 | Linear probe | 冻结 encoder，训练一个轻量分类头。 | linear probe accuracy |
| 25 | 2026-08-02 周日 | Baseline | 做一个简单 baseline: random init encoder 或 MAE-style toy baseline。 | baseline 对比表 |
| 26 | 2026-08-03 周一 | Ablation 1 | 改 mask ratio 或 target block size。 | 一张 ablation 表 |
| 27 | 2026-08-04 周二 | Representation probe | 可选: 用 t-SNE/UMAP 或 nearest neighbor 看 latent。 | latent 可视化或最近邻图 |
| 28 | 2026-08-05 周三 | 第 4 周复盘 | 写 mini-I-JEPA 实验总结。 | `mini_ijepa_report.md` |

理解检查:

- loss 下降是否等于 representation 好？
- linear probe 为什么比直接看 reconstruction 更适合这个任务？
- 如果 mask ratio 变大，模型更难还是更语义化？

## 第 5 周: 从 I-JEPA 过渡到 V-JEPA

目标: 理解视频 JEPA 中时序预测、tubelet、motion representation 的核心变化。

| Day | 日期 | 主题 | 今日任务 | 今日产出 |
|---|---|---|---|---|
| 29 | 2026-08-06 周四 | V-JEPA 论文入口 | 阅读 V-JEPA 摘要、Introduction、Method 图。 | 5 句话解释 V-JEPA 相比 I-JEPA 多了什么 |
| 30 | 2026-08-07 周五 | Video tokens | 学 video patch、tubelet embedding、temporal position。 | 画出 video -> tubelets -> tokens 流程 |
| 31 | 2026-08-08 周六 | Temporal mask | 比较 spatial mask、temporal mask、spatio-temporal mask。 | 三种 mask 的用途表 |
| 32 | 2026-08-09 周日 | Motion vs appearance | 分析 V-JEPA 为什么能兼顾动作和外观任务。 | 写出 motion probe 和 appearance probe 的区别 |
| 33 | 2026-08-10 周一 | 代码阅读 | 浏览 V-JEPA 官方仓库，定位 data、masks、models、evals。 | `vjepa_code_map.md` |
| 34 | 2026-08-11 周二 | Toy video 数据 | 选择 Moving MNIST、合成小球、或者简单网格视频。 | 生成 100 到 1000 条 toy videos |
| 35 | 2026-08-12 周三 | 第 5 周复盘 | 设计 mini-V-JEPA 实验方案。 | `mini_vjepa_design.md` |

理解检查:

- 视频 JEPA 预测未来，是否一定要预测像素？
- motion representation 和 object representation 可能冲突吗？
- 为什么短 horizon 和长 horizon 的困难不一样？

## 第 6 周: 实现 mini-V-JEPA

目标: 跑通 toy video 的 latent prediction，并验证 latent 是否含有运动信息。

| Day | 日期 | 主题 | 今日任务 | 今日产出 |
|---|---|---|---|---|
| 36 | 2026-08-13 周四 | Video dataloader | 写 video dataset，输出形状如 `[B, T, C, H, W]`。 | batch shape 和视频样本可视化 |
| 37 | 2026-08-14 周五 | Video encoder | 先用简化方案: 每帧 CNN/ViT 编码，再加 temporal transformer。 | 前向传播输出 `z_t` |
| 38 | 2026-08-15 周六 | Video masking | 实现遮挡未来帧或部分 tubelet。 | mask 可视化 |
| 39 | 2026-08-16 周日 | Latent predictor | 输入前几帧 latent，预测未来 latent。 | 一次 train step 跑通 |
| 40 | 2026-08-17 周一 | 训练 | 训练 mini-V-JEPA，记录 loss。 | loss 曲线 |
| 41 | 2026-08-18 周二 | Probe | 训练轻量 probe 预测速度、方向或动作标签。 | probe accuracy 或 regression error |
| 42 | 2026-08-19 周三 | 第 6 周复盘 | 写 mini-V-JEPA 实验总结。 | `mini_vjepa_report.md` |

理解检查:

- 你的 latent 预测学到了运动，还是只记住了背景？
- 如果未来有多种可能，单一 MSE latent prediction 会遇到什么问题？
- 视频 JEPA 和 world model 的差别还差在哪里？

## 第 7 周: JEPA 接入 world model

目标: 把 representation learning 变成 action-conditioned latent dynamics。

| Day | 日期 | 主题 | 今日任务 | 今日产出 |
|---|---|---|---|---|
| 43 | 2026-08-20 周四 | World model 模板 | 学 `o_t -> z_t`, `z_t, a_t -> z_{t+1}`。 | 一张 latent world model 架构图 |
| 44 | 2026-08-21 周五 | V-JEPA 2 | 阅读 V-JEPA 2 摘要和 planning 相关部分。 | 写出 V-JEPA 2 如何从视频走向机器人规划 |
| 45 | 2026-08-22 周六 | Toy environment | 选择 gridworld 或 two-room 图像环境。 | 生成图像观测、动作、下一状态数据 |
| 46 | 2026-08-23 周日 | Action-conditioned predictor | 实现 `z_t + action -> z_{t+1}`。 | 一次 action-conditioned train step |
| 47 | 2026-08-24 周一 | Multi-step rollout | 用模型递推预测多步 latent。 | rollout error 随 horizon 变化曲线 |
| 48 | 2026-08-25 周二 | Goal latent | 将目标图像编码成 `z_goal`，定义 latent distance。 | goal-conditioned cost function |
| 49 | 2026-08-26 周三 | 第 7 周复盘 | 整理 world model 失败原因。 | `world_model_review.md` |

理解检查:

- 只有 observation prediction，不输入 action，为什么不能直接用于控制？
- latent distance 和真实任务距离什么时候一致，什么时候不一致？
- rollout horizon 变长后错误为什么会累积？

## 第 8 周: Planning、课题包装和实习表达

目标: 做出可展示 demo，并把它包装成课题和实习能力。

| Day | 日期 | 主题 | 今日任务 | 今日产出 |
|---|---|---|---|---|
| 50 | 2026-08-27 周四 | CEM planner | 学 CEM: 采样动作序列、rollout、打分、保留 top-k、重采样。 | CEM 伪代码 |
| 51 | 2026-08-28 周五 | Planner 实现 | 在 learned latent model 上实现 CEM 或枚举规划。 | planner 能输出动作序列 |
| 52 | 2026-08-29 周六 | 对比实验 | 比较 random policy、oracle shortest path、learned latent planner。 | success rate 对比表 |
| 53 | 2026-08-30 周日 | Ablation | 改 planning horizon、sample 数、latent dim、loss。 | planning ablation 表 |
| 54 | 2026-08-31 周一 | Demo 整理 | 保存可视化: 起点、目标、模型规划路径、失败案例。 | demo 图或短视频 |
| 55 | 2026-09-01 周二 | 课题包装 | 写一页研究方向: JEPA-style latent world model for planning。 | `research_pitch.md` |
| 56 | 2026-09-02 周三 | 实习表达 | 写简历 bullet 和面试讲稿。 | `internship_pitch.md` 和 2 分钟讲稿 |

理解检查:

- 为什么 MPC 通常每一步重新规划，而不是一次规划到底？
- CEM 比 random search 多了什么？
- 如果 planner 失败，是 representation、dynamics、cost，还是 search 的问题？

## 每周里程碑

| 周 | 截止日期 | 必须完成 | 判断标准 |
|---|---|---|---|
| 1 | 2026-07-15 | JEPA 直觉笔记 | 能讲清 JEPA 和 MAE 的区别 |
| 2 | 2026-07-22 | I-JEPA 训练伪代码 | 能解释 mask、EMA、predictor、loss |
| 3 | 2026-07-29 | mini-I-JEPA train step | 能完成一次 forward/backward |
| 4 | 2026-08-05 | mini-I-JEPA report | 有 loss、probe、ablation |
| 5 | 2026-08-12 | mini-V-JEPA 设计 | 明确 toy video、mask、probe |
| 6 | 2026-08-19 | mini-V-JEPA report | 有视频 latent prediction 结果 |
| 7 | 2026-08-26 | latent world model | 能做 action-conditioned rollout |
| 8 | 2026-09-02 | planning demo 和 pitch | 有 demo、对比实验、实习表达 |

## 最小可行版本

如果时间不够，保留这些任务:

1. Day 1 到 Day 7: 建立 JEPA 直觉。
2. Day 15 到 Day 21: 跑通 mini-I-JEPA train step。
3. Day 24: 做 linear probe。
4. Day 43 到 Day 49: 做 action-conditioned latent world model。
5. Day 50 到 Day 56: 做 CEM/MPC planning demo 和实习表达。

可以暂时跳过:

- 大规模 V-JEPA 训练。
- 分布式训练。
- 复杂机器人仿真。
- 追求论文级指标。
- 过早阅读所有 related work。

## 课题方向模板

暂定题目:

```text
面向目标条件规划的 JEPA 式 latent world model 研究
```

核心问题:

```text
如何通过自监督 latent prediction 学到既有语义信息、又适合多步规划的状态表示？
```

技术路线:

```text
图像/视频观测 -> JEPA encoder -> latent representation
latent + action -> action-conditioned predictor -> future latent rollout
goal image -> goal latent
planner -> search actions minimizing latent goal distance
```

可做创新点:

1. 表示层面: 让 latent 对 planning 更友好，而不只是对分类友好。
2. 预测层面: 改进多步 rollout 的误差累积。
3. 目标层面: 设计更可靠的 latent distance 或 learned cost。
4. 数据层面: 用无动作视频预训练，再用少量交互数据适配。
5. 应用层面: 从 toy environment 迁移到机器人、无人机、交通或无线网络。

## 实习表达模板

简历 bullet:

```text
Built a JEPA-style latent world model for goal-conditioned planning, combining self-supervised representation learning, action-conditioned latent dynamics, and CEM/MPC planning in a toy visual control environment.
```

2 分钟讲稿结构:

```text
我关注的问题:
传统 pixel prediction 或 model-free RL 的局限:
我采用的 JEPA-style latent prediction:
如何接入 action-conditioned world model:
如何用 CEM/MPC 做 goal-conditioned planning:
实验结果和失败案例:
下一步如何扩展到真实机器人或复杂环境:
```

## 每天不要犯的错误

1. 不要只读论文不画图。
2. 不要只训练 loss 不做 probe。
3. 不要只做 representation，不接 planning。
4. 不要过早追求大模型复现。
5. 不要把 latent distance 默认等同于任务成功，需要实验验证。
