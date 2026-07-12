<!-- fullWidth: false tocVisible: false tableWrap: true -->
# Day 3: MAE vs I-JEPA——从像素重建到表征预测

日期: **2026-07-11 周六**  
主题: **比较 MAE 的生成式重建目标与 I-JEPA 的非生成式 latent prediction 目标**  
前置笔记:

- [Day 1: I-JEPA 必读部分提取笔记](day01_what_is_JEPA.md)
- [Day 2: 自监督学习版图](day02_self_supervised__learning.md)

参考资料:

- MAE: *Masked Autoencoders Are Scalable Vision Learners*  
  <https://arxiv.org/abs/2111.06377>
- I-JEPA: *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*  
  <https://arxiv.org/abs/2301.08243>

> 今天只比较一个核心差异：**面对同一个不可见区域，MAE 预测它的原始像素，I-JEPA 预测它的抽象表征。**

---

## 0. 今天只抓一个问题

**MAE 和 I-JEPA 都遮住图像的一部分，为什么它们不是同一种方法？**

最短答案：

```text
MAE：    visible patches -> encoder -> decoder -> masked pixels
I-JEPA： context patches -> encoder -> predictor -> target representations
```

真正决定二者差异的不是“有没有 mask”，而是：

```text
训练目标位于哪个空间？
```

| 方法 | 预测目标 | Loss 所在空间 | 模型被迫保留的信息 |
|---|---|---|---|
| MAE | 被遮挡 patch 的真实像素 | Pixel space | 颜色、纹理、边缘、局部结构以及语义线索 |
| I-JEPA | target encoder 输出的目标区域表征 | Representation space | target encoder 保留的、可由 context 预测的抽象信息 |

一句话记忆：

> **Mask 决定模型看不到什么，target space 决定模型必须学会什么。**

---

## 1. 两种方法的共同起点

MAE 和 I-JEPA 具有相同的大方向：

1. 都从没有人工标签的图像中构造训练任务。
2. 都只让输入分支看到部分图像内容。
3. 都要求模型利用可见 context 推断不可见区域。
4. 都可以使用 ViT 作为 encoder。
5. 预训练结束后都主要希望得到一个可迁移的视觉 encoder。

因此，从非常高的层次看，二者都在做：

```text
部分观测
   ↓
编码上下文
   ↓
预测缺失信息
```

但“缺失信息”有两种定义：

```text
MAE 认为缺失信息 = 原始像素
I-JEPA 认为缺失信息 = 目标区域的 latent representation
```

这一个差异会继续影响：

- decoder 或 predictor 的职责；
- loss 的含义；
- 模型如何处理不确定性；
- 模型关注低层细节还是抽象语义；
- 预训练任务与 world model 的距离。

---

## 2. 先统一符号

设输入图像被切成 `N` 个 patch：

```math
x = \{x_1,x_2,\ldots,x_N\}
```

对 MAE：

- `V`：visible patch positions；
- `M`：masked patch positions；
- `V ∪ M` 覆盖全部 patch positions；
- `V ∩ M = ∅`。

对 I-JEPA：

- `C`：context positions；
- `T_1, T_2, ..., T_K`：一个或多个 target blocks；
- context 中会移除与 target 重叠的内容，避免直接看到答案。

后面所有流程图都围绕这几个集合展开。

---

## 3. MAE 完整流程图：从 visible patches 重建 pixels

### 3.1 今日产出一：MAE 流程图

```text
                         原始完整图像 x
                                │
                                ▼
                         Patchify + position
                                │
                                ▼
                    随机遮挡约 75% patches
                         │               │
                         │               └───────────────┐
                         ▼                               │
                  Visible patches x_V                   │
                         │                               │
                         ▼                               │
                 ViT Encoder f_theta                    │
              （只处理 visible tokens）                 │
                         │                               │
                         ▼                               │
             Visible latent representations h_V         │
                         │                               │
                         ▼                               │
       投影到 decoder 维度，并在 masked positions        │
       插入 learned mask tokens + decoder position       │
                         │                               │
                         ▼                               │
                 Lightweight Decoder d_phi              │
                         │                               │
                         ▼                               │
              所有位置的 pixel predictions x_hat        │
                         │                               │
                 只选择 masked positions                 │
                         │                               │
                         ▼                               ▼
                 predicted pixels x_hat_M  与  true pixels x_M
                                │
                                ▼
                      Pixel reconstruction MSE
```

需要立刻看懂的三个点：

1. Encoder 只接收真实可见 patches，不接收 mask tokens。
2. Decoder 会看到完整位置序列：visible representations 加 masked positions 的占位 token。
3. 真实 masked pixels 不进入模型输入，只在最后作为 loss target。

### 3.2 用一个 `4 × 4` patch 网格理解

原图被分成 16 个 patch：

```text
 1   2   3   4
 5   6   7   8
 9  10  11  12
13  14  15  16
```

假设可见位置是 `1、4、7、13`：

```text
 V   M   M   V
 M   M   V   M
 M   M   M   M
 V   M   M   M
```

Encoder 输入只有：

```text
patch_1 + position_1
patch_4 + position_4
patch_7 + position_7
patch_13 + position_13
```

Encoder 输出：

```text
h_1, h_4, h_7, h_13
```

Decoder 输入会恢复成完整的 16 个位置：

```text
 h_1    mask_2   mask_3    h_4
mask_5  mask_6    h_7     mask_8
mask_9  mask_10  mask_11  mask_12
 h_13   mask_14  mask_15  mask_16
```

这里的 `mask_i` 可以理解为：

```text
learned_mask_token + positional_embedding_i
```

它只告诉 decoder：

> “位置 `i` 有一块未知内容，请根据其他可见信息预测它。”

Mask token 不包含真实像素，不会泄露答案。

### 3.3 为什么只 encode visible patches 仍能重建

因为 encoder 和 decoder 分工不同：

```text
Encoder：从可见内容中提取上下文线索
Decoder：根据线索和缺失位置完成像素填空
```

自然图像具有结构和统计规律：

- 同一物体的轮廓通常连续；
- 身体部件之间存在空间关系；
- 邻近区域的颜色和纹理具有相关性；
- 天空、道路、建筑等具有常见布局；
- 某些物体经常在同一场景中出现。

所以模型可以从：

```text
狗头 + 一条腿 + 草地
```

推断缺失区域很可能包含：

```text
狗的身体、其他腿和连续的草地背景
```

但“可以预测”不等于“能够从信息上精确找回”。若缺失区域中有完全不可预测的毛发、草叶或噪声，模型只能根据训练分布给出统计上合理的结果。

### 3.4 MAE 的 loss

对所有 masked positions `M`，MAE 的学习目标可以写成：

```math
L_{MAE}
= \frac{1}{|M|}
  \sum_{p \in M}
  ||\hat{x}_p-x_p||_2^2
```

其中：

- `x_p`：原图位置 `p` 的真实 patch pixels；
- `x_hat_p`：decoder 的 pixel prediction；
- loss 通常只在 masked positions 上计算；
- 有些配置会先对每个 target patch 的像素进行归一化。

### 3.5 MAE 的梯度路径

```text
Pixel MSE
   │
   ▼
Decoder parameters phi
   │
   ▼
Visible latent representations
   │
   ▼
Encoder parameters theta
```

真实像素 `x_M` 是固定 target，不需要更新。Loss 通过 decoder 反向传播到 encoder，迫使 encoder 产生有利于像素重建的可见表征。

### 3.6 为什么采用重型 encoder + 轻型 decoder

MAE 的目标是训练一个以后可以迁移的 encoder，而不是保存一个昂贵的图像生成器。

当遮挡率为 75% 时，encoder 只处理约 25% 的 tokens，可以显著减少 Transformer self-attention 的计算量。之后用较轻的 decoder 处理完整序列并完成预训练任务。

```text
训练阶段：heavy encoder + lightweight decoder
下游阶段：保留 encoder，通常丢弃 reconstruction decoder
```

---

## 4. I-JEPA 完整流程图：从 context 预测 target representations

### 4.1 今日产出二：I-JEPA 流程图

```text
                               同一张完整图像 x
                         ┌────────────┴────────────┐
                         │                         │
                         ▼                         ▼
              采样 context 与 target blocks    Target Encoder f_xi
                         │                   （处理完整图像）
              移除 context-target 重叠             │
                         │                         ▼
                         ▼                 全图 patch representations s
                 Context patches x_C               │
                         │                  选择 target positions T
                         ▼                         │
              Context Encoder f_theta              ▼
                         │                 target representations s_T
                         ▼                         │
              context representations h_C       stop-gradient
                         │                         │
                         ├──────────────┐          │
                         │              │          │
                         ▼              ▼          │
                 target position   learned target  │
                    information       tokens       │
                         └──────┬───────┘          │
                                ▼                  │
                         Predictor q_psi           │
                                │                  │
                                ▼                  │
                predicted target representations s_hat_T
                                │                  │
                                └────────┬─────────┘
                                         ▼
                         Representation-space L1 loss

              Target encoder 不接收反向传播梯度：
              xi <- momentum * xi + (1-momentum) * theta
```

需要立刻看懂的四个点：

1. Context encoder 只看到 context，不能直接读取 target 内容。
2. Target encoder 处理完整图像，再从输出中选择 target positions 的 representations。
3. Predictor 预测的不是 RGB，而是 target encoder 的 latent representations。
4. Target branch 使用 stop-gradient，target encoder 通过 context encoder 的 EMA 更新。

### 4.2 I-JEPA 的 context 和 target 怎么构造

I-JEPA 不是简单地独立随机遮掉许多小 patch。它通常采样：

- 一个较大的 context block；
- 多个具有一定空间范围的 target blocks；
- 从 context 中移除与 targets 重叠的部分。

直观示意：

```text
┌──────────────────────────────┐
│          context 可见区域     │
│                              │
│      ┌──────────┐            │
│      │ target 1 │            │
│      └──────────┘            │
│                    ┌───────┐ │
│                    │target2│ │
│                    └───────┘ │
└──────────────────────────────┘

送入 context encoder 前，target 位置会从 context 中去掉。
```

使用较大的连续 target blocks，是为了避免任务退化成只根据邻近纹理补一个小洞。目标区域越大，模型越需要使用长距离结构和语义信息。

I-JEPA 的 multi-block masking 原图：

![Figure 4: I-JEPA multi-block masking](assets/figure4_masking.png)

### 4.3 Target encoder 到底提供什么

Target encoder 给出的是目标区域的参考表征：

```math
s = f_\xi(x)
```

然后从全图 representations 中选出 target positions：

```math
s_T = select_T(s)
```

这些 `s_T` 不是人工标签，也不是固定类别，而是 target encoder 当前认为目标区域应该具有的 latent representations。

因此，I-JEPA 的监督信号来自同一张图像内部：

```text
Context branch 尝试预测 target
Target branch 产生当前训练目标
```

### 4.4 Predictor 做什么

Predictor 接收：

- context representations；
- target positions；
- 用于表示未知 target content 的 learned tokens。

输出：

```text
每个 target position 的 predicted representation
```

可以写成：

```math
\hat{s}_T = q_\psi(h_C, pos_T)
```

其中：

```math
h_C = f_\theta(x_C)
```

Predictor 不需要生成可以直接显示的图片。它只需要在 target encoder 定义的 latent space 中预测正确位置的表示。

### 4.5 I-JEPA 的 loss

如果共有 `K` 个 target blocks，每个 block 内又包含若干 target patches，可以把 loss 简化写成：

```math
L_{I-JEPA}
= \frac{1}{K}
  \sum_{k=1}^{K}
  \frac{1}{|T_k|}
  \sum_{p \in T_k}
  ||\hat{s}_p-stopgrad(s_p)||_1
```

最关键的部分是：

```text
比较 s_hat_p 和 s_p
而不是比较 x_hat_p 和 x_p
```

### 4.6 I-JEPA 的梯度与 EMA 路径

反向传播路径：

```text
Representation L1 loss
          │
          ▼
     Predictor q_psi
          │
          ▼
Context Encoder f_theta
```

Target branch：

```text
Target Encoder f_xi
          │
          ▼
target representations
          │
       stop-gradient
```

Target encoder 不通过 loss 直接更新，而是使用 context encoder 参数的指数移动平均：

```math
\xi \leftarrow m\xi+(1-m)\theta
```

其中 `m` 接近 1，使 target representations 随训练平滑变化。

完整 I-JEPA 训练结构原图：

![Figure 3: I-JEPA training structure](assets/figure3_ijepa.png)

---

## 5. 两张流程图并排压缩

| MAE | I-JEPA |
|---|---|
| `image -> random patch mask` | `image -> context + target block sampling` |
| `visible patches -> encoder` | `context patches -> context encoder` |
| `visible latents + mask tokens -> decoder` | `context latents + target positions -> predictor` |
| `decoder -> predicted target pixels` | `predictor -> predicted target representations` |
| `original image -> true target pixels` | `full image -> EMA target encoder -> true target representations` |
| `pixel MSE` | `representation L1` |
| Loss 反传到 decoder 和 encoder | Loss 反传到 predictor 和 context encoder |
| 真实像素是固定 target | Target representation 随 EMA teacher 缓慢变化 |

最短流程：

```text
MAE
visible pixels -> encoder -> decoder -> predicted pixels
                                      ↕ MSE
                                  original pixels
```

```text
I-JEPA
context -> context encoder -> predictor -> predicted target embedding
                                              ↕ L1
full image -> EMA target encoder -------> target embedding
```

---

## 6. 真正的核心：Pixel space 与 representation space

### 6.1 Pixel target 要求模型解释什么

一个 patch 的像素包含：

- 物体类别和结构；
- 颜色与光照；
- 材料纹理；
- 阴影和反射；
- 背景细节；
- 传感器噪声；
- 许多偶然的高频变化。

MAE 的 pixel loss 不会自动区分：

```text
哪些信息对语义重要
哪些信息只是低层、偶然或不可预测的细节
```

为了降低像素误差，模型需要尽量把它们都考虑进去。

### 6.2 Latent target 尝试保留什么

Target encoder 可以把很多像素级变化映射到相近的 latent representation。例如：

```text
同一类物体在不同光照下的像素差异很大
但其高层结构和语义表征可能相近
```

因此 I-JEPA 希望 predictor 重点预测：

- 物体结构；
- 场景组成；
- 空间关系；
- target encoder 认为稳定的抽象属性；
- 能够从 context 合理推断的部分。

需要保持谨慎：

> Latent 并不天然等于“完美语义”。Target encoder 学到什么仍取决于数据、mask、架构、优化和防坍塌机制。

### 6.3 “预测容易”不等于“任务简单”

I-JEPA 放弃像素级目标，不代表它只做一个简单任务。

如果 target block 足够大，而 context 又不能直接看到 target，模型必须通过：

- 全局场景理解；
- 对象部件关系；
- 长距离依赖；
- 空间位置信息；
- 抽象语义一致性；

才能预测 target representation。

所以更准确的说法是：

```text
I-JEPA 不是简单降低任务难度，
而是把难度从“恢复所有像素细节”
转移到“预测可抽象、可迁移的目标表征”。
```

---

## 7. 用同一个遮挡区域理解差别

假设图片是一只狗站在草地上，狗的身体区域被遮挡，只能看到：

```text
狗头 + 一条腿 + 一部分尾巴 + 草地
```

### 7.1 MAE 面对的问题

MAE 要回答：

```text
狗身体区域每一个 patch 的 RGB 像素应该是多少？
```

它需要处理：

- 毛发颜色；
- 每根毛发的方向；
- 阴影；
- 草叶纹理；
- 身体轮廓；
- 光照强弱。

其中很多细节仅凭 context 并不能唯一确定。

### 7.2 I-JEPA 面对的问题

I-JEPA 要回答：

```text
这个 target region 在 target encoder 的 latent space 中应该是什么？
```

即使不能确定每根毛发的像素，也可能预测出：

```text
这是狗的身体部分
它与可见狗头和腿属于同一对象
它位于当前目标位置
它应具有与动物身体相关的结构表征
```

### 7.3 多种合理答案带来的问题

假设相同 context 下，target pixels 可能有多个合理版本：

```text
版本 A：毛发向左，光线较亮
版本 B：毛发向右，光线较暗
版本 C：毛发纹理更密，带有小块阴影
```

在像素空间，它们差异明显；在抽象表征空间，它们可能都表示：

```text
狗的身体区域
```

因此：

```text
Pixel prediction 需要选择或平均许多低层可能性
Latent prediction 可以让多个低层结果共享一个抽象目标
```

---

## 8. 为什么 MSE 可能产生“平均结果”

这是理解 JEPA 动机的重要一步。

假设给定相同 context，真实 target 可能是两个值：

```text
可能结果 1：-1
可能结果 2：+1
```

如果两种结果概率相同，而模型使用 MSE，只能输出一个确定值 `y_hat`，那么使平均平方误差最小的结果是：

```text
y_hat = 0
```

但 `0` 可能不是任何一个真实模式。

图像中类似现象可能表现为：

- 多种可能纹理被平均成模糊纹理；
- 多种边缘位置被平均成柔和边缘；
- 多个合理细节被压成不够真实的中间结果。

数学上，对给定 context `c`，MSE 的最优确定性预测倾向于条件均值：

```math
\hat{y}^*(c)=E[Y\mid C=c]
```

I-JEPA 的思路不是直接解决所有生成分布问题，而是先通过 target encoder 抽象掉一部分难以预测的低层变化，再预测更稳定的 latent target。

---

## 9. “Non-generative” 到底是什么意思

I-JEPA 被称为非生成式目标，不是说它完全不做 prediction，而是说：

```text
它不要求在原始数据空间中生成目标 y
```

MAE 比较：

```text
D(x_hat_target, x_target)
```

I-JEPA 比较：

```text
D(s_hat_target, s_target)
```

其中：

- `x_target` 是原始 pixels；
- `s_target` 是 target encoder 产生的 representation。

因此：

```text
MAE 输出可以被还原为可观看的像素 patch
I-JEPA predictor 输出通常只是 latent vectors，不能直接当图片观看
```

I-JEPA 仍然在预测，只是预测对象从“数据本身”换成了“数据的抽象表示”。

---

## 10. 两种方法中的“答案”分别从哪里来

### 10.1 MAE 的答案

```text
答案来源：原始完整图像
答案形式：masked positions 的真实 RGB pixels
答案是否随训练变化：否
```

原始图像对某个 patch 的像素是固定的。

### 10.2 I-JEPA 的答案

```text
答案来源：EMA target encoder
答案形式：target positions 的 latent representations
答案是否随训练变化：是，但通过 EMA 平滑变化
```

这意味着 I-JEPA 的 target 不是预先固定的。Encoder 学习的同时，target representation 也在逐渐演化。

这带来两个结果：

- 优点：目标可以随着模型学习逐渐形成抽象表示；
- 风险：如果目标分支退化，可能出现 representation collapse。

这正是 Day 4 要继续研究的问题。

---

## 11. 模块职责对照

| 模块职责 | MAE | I-JEPA |
|---|---|---|
| 提取可见信息 | Encoder | Context encoder |
| 表示未知目标位置 | Mask token + decoder position | Target token/position information |
| 产生预测 | Reconstruction decoder | Predictor |
| 提供真实目标 | 原始图像 pixels | EMA target encoder representations |
| 预测输出 | RGB pixel values | Latent vectors |
| 直接接收梯度的预测模块 | Decoder | Predictor |
| 直接接收梯度的主干 | Encoder | Context encoder |
| 不接收梯度的目标分支 | 原始 pixels 本来就是常量 | Target encoder 使用 stop-gradient |
| 下游主要保留 | Encoder | Pretrained encoder |

不要把下面三组概念混为一谈：

```text
MAE decoder ≠ I-JEPA predictor
MAE original pixels ≠ I-JEPA target encoder representations
MAE mask tokens ≠ target 内容本身
```

---

## 12. Masking strategy 也不完全相同

虽然 target space 是最核心差异，但 masking strategy 也不同。

### MAE 的典型 masking

```text
在所有 patch positions 中进行高比例随机采样
常用约 75% mask ratio
masked patches 比较分散
```

目的包括：

- 让重建任务具有足够难度；
- 降低 encoder token 数量；
- 避免只依赖局部复制。

### I-JEPA 的典型 masking

```text
采样多个相对较大的连续 target blocks
再采样较大的 context block
从 context 中删除与 targets 重叠的区域
```

目的包括：

- 让 target 更接近语义级区域，而不是孤立小纹理；
- 让 context 保留足够信息，使任务可预测；
- 迫使模型利用长距离信息预测目标表征。

因此不能简单理解成：

```text
I-JEPA = 把 MAE 的 decoder 换成 predictor
```

更完整的变化是：

```text
目标空间改变
+ target encoder
+ predictor
+ EMA/stop-gradient
+ multi-block masking strategy
```

---

## 13. 哪一种方法更好

不能脱离任务给出绝对排名。

### MAE 更有吸引力的情况

- 希望保留细粒度局部结构；
- 下游任务依赖纹理、边缘和空间细节；
- 需要一个结构简单、扩展性强的 ViT 预训练目标；
- 有兴趣研究像素重建或生成相关任务。

### I-JEPA 更有吸引力的情况

- 更关心抽象语义和可预测状态；
- 不希望模型把大量容量用于不可预测的像素细节；
- 希望避免对比学习中的大规模负样本；
- 希望进一步连接视频表征、latent dynamics 和 world model。

### 公平结论

```text
MAE 的 inductive pressure：恢复数据细节
I-JEPA 的 inductive pressure：预测抽象目标状态
```

最终仍需要根据具体数据、模型规模和下游任务进行实证评估。不能仅凭“latent 比 pixel 更抽象”就断言 I-JEPA 在所有任务上都更好。

---

## 14. 为什么 I-JEPA 更接近 world model

一个 latent world model 通常需要：

```text
当前观测 o_t
   -> encoder
   -> latent state s_t

latent state s_t + action a_t
   -> dynamics/predictor
   -> predicted future latent state s_hat_{t+1}
```

I-JEPA 已经具备类似的抽象形式：

```text
context representation
   -> predictor
   -> target representation
```

而 MAE 的目标仍是：

```text
visible observations
   -> decoder
   -> missing raw observations
```

因此 I-JEPA 在“预测 latent state”这一点上更接近 world model。

但图像版 I-JEPA 仍然缺少：

- 明确的时间方向；
- action conditioning；
- 多步动力学；
- reward 或 cost；
- planner。

所以准确说法是：

> I-JEPA 提供了更接近 latent world model 的训练思想，但它本身不是完整的 action-conditioned world model。

---

## 15. 六个理解检查

### 1. MAE 为什么只把 visible patches 送进 encoder

因为被遮挡位置没有真实输入内容，大量 mask tokens 进入重型 encoder 会浪费计算。Encoder 只提取 visible context，轻量 decoder 再结合 mask tokens 和位置信息预测缺失像素。

### 2. Decoder 如何知道需要预测哪个位置

每个 mask token 都与对应位置的 positional embedding 结合，因此 decoder 知道未知 token 位于图像的哪个 patch position。

### 3. I-JEPA 不比较 pixels，怎么知道预测是否正确

Target encoder 处理完整图像，并从输出中选择 target positions 的 representations。Predictor 的输出与这些 stop-gradient target representations 计算 L1 loss。

### 4. MAE 和 I-JEPA 最核心的区别是什么

不是二者是否使用 mask，而是 target space：MAE 预测 pixel target，I-JEPA 预测 learned representation target。

### 5. 为什么 latent prediction 可能减少不可预测细节的影响

Target encoder 可以把多种低层像素变化映射到相近的抽象表示，使 predictor 不必恢复每个纹理、光照和噪声细节。

### 6. Latent 是否天然等于语义

不是。Latent 的内容取决于 encoder 的学习过程。Masking、teacher 更新、架构和防坍塌机制共同影响它最终保留什么信息。

---

## 16. 常见误区

### 误区 1：MAE 的 encoder 会直接输出完整图像

错误。Encoder 只输出 visible patches 的 latent representations；decoder 才产生像素预测。

### 误区 2：MAE 的 mask token 中藏着真实像素

错误。Mask token 只是 learned placeholder，真实像素只在 loss 端提供。

### 误区 3：I-JEPA 的 target encoder 只输入裁剪后的 target blocks

不准确。标准理解是 target encoder 处理完整图像，然后在输出 representation grid 中选取 target positions。

### 误区 4：I-JEPA 不生成像素，所以没有训练目标

错误。它有明确 target，只是 target 位于 learned representation space。

### 误区 5：I-JEPA 可以直接显示预测图片

通常不可以。Predictor 输出 latent vectors；若要变成像素，还需要额外训练 image decoder。

### 误区 6：预测 latent 一定不会 collapse

错误。若所有输入和 target 都变成相同向量，latent loss 也可能很小。为什么训练没有走向这个平凡解，需要结合 EMA、stop-gradient、predictor 和整体训练动力学继续分析。

### 误区 7：I-JEPA 只是换了一种 loss

不完整。它同时改变了 target encoder、梯度路径、predictor、masking strategy 和预测空间。

---

## 17. 不看笔记时应该能复述的版本

### 17.1 30 秒版本

> MAE 和 I-JEPA 都从部分可见图像预测不可见区域，但目标空间不同。MAE 的 encoder 只处理可见 patches，decoder 插入 mask tokens 并重建缺失区域的原始像素，使用 masked pixel MSE。I-JEPA 的 context encoder 处理 context，predictor 根据 context 和 target 位置预测 latent representations；target 由处理完整图像的 EMA target encoder 提供，loss 是 representation-space L1。MAE 更强调恢复像素细节，I-JEPA 希望预测可由上下文推断的抽象信息。

### 17.2 三行记忆版

```text
MAE：    context -> encoder -> decoder -> pixels
I-JEPA： context -> encoder -> predictor -> latent targets
核心：   pixel reconstruction vs representation prediction
```

### 17.3 一句主线

> **MAE 要回答“缺失区域长什么样”，I-JEPA 要回答“缺失区域在抽象空间中是什么”。**

---

## 18. 今日完成标准

- [ ] 能独立画出 MAE 的训练流程图。
- [ ] 能独立画出 I-JEPA 的双分支训练流程图。
- [ ] 能说明 MAE encoder 为什么只处理 visible patches。
- [ ] 能指出真实 pixels 在 MAE 中只在哪里出现。
- [ ] 能指出 I-JEPA 的 target representations 从哪里产生。
- [ ] 能画出两种方法各自的梯度传播路径。
- [ ] 能解释 pixel MSE 面对多种合理结果时为什么可能产生平均预测。
- [ ] 能说明“latent 更抽象”不等于“必然更好”。

---

## 19. 与 Day 4 的衔接：表征坍塌

今天最后留下一个问题：

```text
如果 I-JEPA 的 context encoder 和 target encoder
对所有图像、所有位置都输出同一个向量，
那么 predicted representation 和 target representation
不是也可以完全相等、loss 等于 0 吗？
```

这就是 representation collapse 的核心担忧。

Day 4 将继续学习：

- 什么是 collapse；
- 为什么只拉近 latent representations 容易出现平凡解；
- online/context encoder 与 target encoder 如何分工；
- stop-gradient 有什么作用；
- EMA target 为什么能提供更稳定的训练目标；
- BYOL、DINO 与 I-JEPA 的相关设计如何理解。

---

## 20. 今日日志模板

```text
日期：2026-07-11
主题：MAE vs I-JEPA

今天最重要的一句话：
Mask 决定模型看不到什么，target space 决定模型必须学会什么。

我已经能独立画出的流程：
1. MAE：
2. I-JEPA：

我现在能解释：
1. MAE 为什么只 encode visible patches：
2. MAE 的 decoder 如何预测 masked positions：
3. I-JEPA 的 target 从哪里来：
4. Pixel prediction 与 latent prediction 的核心差异：

我还不清楚：
1.
2.

今天的产出文件：
notes/day03_mae_vs_ijepa.md

下一步：
学习 representation collapse、stop-gradient 与 EMA target encoder。
```
