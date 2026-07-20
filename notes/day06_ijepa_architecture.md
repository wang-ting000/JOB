<!-- fullWidth: false tocVisible: false tableWrap: true -->
# Day 6: I-JEPA 架构拆解——从 context tokens 到 target representations

日期: **2026-07-14 周二**  
主题: **拆解 context encoder、target encoder、predictor 与 mask sampler，并追踪数据、梯度和参数更新路径**  
前置笔记:

- [Day 3: MAE vs I-JEPA](day03_mae_vs_ijepa.md)
- [Day 4: 表征坍塌](day04_representation_collapse.md)
- [Day 5: ViT 基础](day05_vit_basics.md)

参考资料:

- I-JEPA: *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*  
  <https://arxiv.org/abs/2301.08243>
- I-JEPA 官方代码  
  <https://github.com/facebookresearch/ijepa>

> 今天只抓一个问题：**I-JEPA 如何只看 context patches，就预测指定 target blocks 的 latent representations？**

---

## 0. 今天的直接答案

I-JEPA 使用两个编码分支和一个预测器：

```text
context patches
    -> context encoder
    -> contextualized visible tokens
    -> predictor + target position tokens
    -> predicted target representations z_pred

完整图像
    -> target encoder
    -> full-image patch representations
    -> 选取 target positions
    -> target representations z_target

loss = average L1(z_pred, stop_gradient(z_target))
```

训练时：

- `context encoder` 和 `predictor` 接收梯度，由 optimizer 更新；
- `target encoder` 不接收梯度；
- `target encoder` 的参数由 `context encoder` 参数的 EMA 更新；
- predictor 知道目标位置，但看不到对应的 target pixels；
- 预训练结束后，主要保留 encoder 作为视觉表征模型。

一句话记忆：

> **Context branch 提供“已知内容”，target branch 生成“表征答案”，predictor 根据目标位置完成 latent-space 填空。**

---

## 0.1 今日学习安排（约 2 小时）

| 时间 | 内容 | 必须留下的产出 |
|---:|---|---|
| 0–15 分钟 | 复习 ViT patch tokens | 写出 `[B,C,H,W] -> [B,N,D]` |
| 15–40 分钟 | 阅读第 1–4 节 | 画出四模块总图 |
| 40–65 分钟 | 阅读第 5–7 节 | 标出每条数据流的 shape |
| 65–85 分钟 | 阅读第 8–10 节 | 标出 gradient、optimizer、EMA |
| 85–105 分钟 | 阅读 mask 与防泄漏部分 | 解释 target 为何要大、context 为何要分散 |
| 105–120 分钟 | 完成练习和日志 | 独立默画一次训练流程 |

如果今天只有 60 分钟：

```text
第 1 步：读第 0、2、3、4 节，认清四个模块。
第 2 步：读第 7、8 节，认清 loss 和更新路径。
第 3 步：完成练习 1，并闭卷画出第 2 节总图。
```

---

## 1. 统一符号和 shape

设输入图像为：

```text
x: [B, C, H, W]
```

patch size 为 `P`，则网格大小和 patch 数为：

```math
H_p=H/P, \qquad W_p=W/P, \qquad N=H_pW_p
```

ViT 将图像转成 `N` 个 `D` 维 patch tokens：

```text
patch embeddings: [B, N, D]
```

定义：

- `C`：context positions，可见 patch 的位置集合；
- `T_k`：第 `k` 个 target block 的位置集合；
- `N_c=|C|`：context token 数；
- `N_t=|T_k|`：某个 target block 的 token 数；
- `K`：一张图像采样的 target blocks 数量。

主要张量可记作：

```text
z_context: [B, N_c, D_e]
z_target:  [B, K, N_t, D_e]  # 为讲解而写的规则化形式
z_pred:    [B, K, N_t, D_e]
```

实际代码中，不同 target block 的 token 数可能不同，也可能把 `K`、batch 或 target tokens 合并到其他维度。**关键不在具体排布，而在参与 loss 的 `z_pred` 与 `z_target` 必须逐位置对齐。**

---

## 2. 四模块总览

| 模块 | 输入 | 输出 | 核心职责 | 是否接收梯度 | 更新方式 |
|---|---|---|---|---|---|
| Mask sampler | patch 网格与采样参数 | context mask、target masks | 决定模型能看什么、要预测什么 | 否 | 无可学习参数 |
| Context encoder | context patches 与位置 | `z_context` | 编码可见上下文 | 是 | Optimizer |
| Target encoder | 完整图像 tokens | full target features | 产生稳定的 latent 监督信号 | 否 | EMA |
| Predictor | `z_context`、context positions、target positions | `z_pred` | 预测指定位置的 target features | 是 | Optimizer |

完整训练流：

```text
                           ┌────────────────────────────┐
                           │       mask sampler         │
                           │ context C + targets T_1..K │
                           └─────────────┬──────────────┘
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             │                                                       │
             v                                                       v
  x 的 context patches                                     完整图像 x
             │                                                       │
             v                                                       v
  context encoder θ                                        target encoder ξ
             │                                                       │
             v                                                       v
       z_context                                      full patch features
             │                                                       │
             │                                               gather T_1..K
             │                                                       │
             v                                                       v
 predictor + target positions ──> z_pred              stop-gradient(z_target)
             │                                                       │
             └────────────────────── L1 loss ────────────────────────┘

optimizer: update θ and predictor parameters
EMA:       ξ <- mξ + (1-m)θ
```

---

## 3. Mask sampler：定义问题，而不只是制造缺失

### 3.1 Target blocks

I-JEPA 在同一图像中采样若干空间连续的 target blocks。模型需要预测这些位置对应的 target-encoder representations。

Target block 需要足够大，原因是：

- 过小的目标可能只需利用局部纹理或边缘连续性；
- 较大的目标更可能覆盖物体或有语义意义的区域；
- 当目标不确定时，latent target 可以忽略难以预测的像素级细节。

### 3.2 Context block

Context 是 target blocks 之外、提供给 context encoder 的可见 patches。它需要包含足够多且空间上分散的信息。

Context 太少时：

```text
输入信息不足 -> target 本身不可预测 -> predictor 只能学平均答案
```

Context 太局部时：

```text
模型只利用邻近纹理 -> 不必形成全局语义理解
```

### 3.3 防止信息泄漏

最重要的不变量是：

```text
context encoder 的输入中不能包含 target pixels。
```

注意区分：

- predictor 可以知道 target **在哪里**；
- predictor 不可以看到 target **是什么像素内容**。

位置不是答案。若没有 target position，面对同一个 `z_context`，predictor 不知道当前应该输出左上角、中心还是右下角的 representation。

---

## 4. Context encoder：把可见 patches 变成上下文化特征

Context encoder 通常是 ViT。它只处理 context positions 对应的 patch tokens：

```text
visible patch embeddings: [B, N_c, D_e]
          + position embeddings
                    │
                    v
             ViT encoder
                    │
                    v
z_context: [B, N_c, D_e]
```

`z_context` 不是 raw pixels，也不是彼此独立的 patch embeddings。经过 self-attention 后，每个 visible token 都融合了其他可见 tokens 的信息。

Context encoder 的职责不是直接输出 target features，而是提供 predictor 可利用的上下文表示。

它接收来自 loss 的梯度：

```text
loss -> predictor -> z_context -> context encoder
```

因此 context encoder 会学习：什么信息最有利于预测同一图像中其他区域的表征。

---

## 5. Target encoder：产生会缓慢变化的学习目标

Target encoder 与 context encoder 使用相同类型的网络结构，但承担不同角色。

它接收完整图像，先生成所有 patch positions 的 contextualized features：

```text
full image
   -> all patch embeddings [B, N, D_e]
   -> target encoder
   -> full target features [B, N, D_e]
   -> gather target positions T_1...T_K
   -> z_target
```

这里有两个容易混淆的点。

### 5.1 Target encoder 看完整图像，不等于 predictor 泄漏

Target branch 的作用是制作训练标签。`z_target` 会经过 stop-gradient，不能把 target pixels 通过梯度路径传给 predictor。

类比监督学习：

```text
分类标签中包含答案，但模型输入中没有标签。
```

同理：

```text
target encoder 知道完整图像，用它生成监督信号；
context encoder 和 predictor 仍然只从 context 内容作答。
```

### 5.2 Target feature 不只是局部 patch 内容

Target encoder 对完整 token 序列进行 self-attention。因此某个 target position 的 feature 通常包含全图上下文，而不是该 patch 的 RGB 压缩版。

这使预测目标更偏向上下文化 representation，而非逐像素复原。

---

## 6. Predictor：根据 context 和目标位置作答

Predictor 的输入包含两类信息：

1. context encoder 输出的 `z_context`；
2. 表示 target positions 的位置 token 或 mask token。

概念流程：

```text
z_context + context position information
                         │
target mask tokens + target position information
                         │
                         v
                 Transformer predictor
                         │
                         v
             target-position outputs z_pred
```

### 6.1 为什么需要 target position

设相同的 context 对应两个目标：

```text
T_1 = 左上角区域
T_2 = 右下角区域
```

如果 predictor 只收到相同的 `z_context`，却没有位置条件，它无法判断要预测哪一个区域。

因此可把 predictor 理解为函数：

```math
z_{pred}=g(z_{context},\ position_{target})
```

### 6.2 Predictor 与 MAE decoder 的区别

| 对比项 | MAE decoder | I-JEPA predictor |
|---|---|---|
| 预测目标 | masked pixels | target representations |
| 输出空间 | RGB / pixel space | latent space |
| 主要压力 | 还原视觉细节 | 预测抽象、上下文化特征 |
| 目标网络 | 不需要 | EMA target encoder |

它们都可接收 mask/position tokens，但“预测空间”不同，因此训练出的 encoder 倾向也不同。

---

## 7. Loss：比较同一目标位置的两个 representations

对每个 target block 及其中每个 target position，I-JEPA 比较预测表示与 target encoder 产生的表示。

简化写法：

```math
\mathcal{L}=\frac{1}{K}\sum_{k=1}^{K}
\frac{1}{|T_k|}\sum_{j\in T_k}
\left\|z^{(k,j)}_{pred}-\operatorname{sg}(z^{(k,j)}_{target})\right\|_1
```

其中：

- `sg` 表示 stop-gradient；
- 比较必须发生在相同 target position；
- `z_pred` 与 `z_target` 的最后一维必须一致；
- 实际实现可能对 batch、target blocks 和 tokens 采用不同的 reshape 或平均次序。

Loss 小只表示预测逐渐接近当前 target representations。它不自动证明 representation 对下游任务有用，所以后面仍需 linear probe、kNN 或 fine-tuning 评估。

---

## 8. 梯度与参数更新：一张必须会画的图

设：

- context encoder 参数为 `θ`；
- target encoder 参数为 `ξ`；
- predictor 参数为 `φ`。

反向传播路径：

```text
                       loss
                      /    \
                     /      X  stop-gradient
                    v        \
             predictor φ     z_target
                    │             │
                    v             │
             context encoder θ    target encoder ξ
```

Optimizer 更新：

```text
θ <- optimizer_step(θ, grad_θ)
φ <- optimizer_step(φ, grad_φ)
```

Target encoder 使用 EMA 更新：

```math
\xi \leftarrow m\xi+(1-m)\theta
```

其中 `m` 接近 1。直觉上：

```text
新的 target 参数
= 大部分旧 target 参数
+ 小部分最新 context 参数
```

### 8.1 更新顺序的概念版

```text
1. forward 得到 z_pred 与 z_target
2. 计算 loss
3. backward，只对 θ 和 φ 求梯度
4. optimizer 更新 θ 和 φ
5. 用更新后的 θ 对 ξ 做 EMA
```

具体工程实现可能在 step 边界处安排 EMA，但必须保持：target encoder 不被 optimizer 直接更新。

### 8.2 为什么不让 target encoder直接追着 loss 跑

如果两边同时通过同一 loss 快速移动：

```text
预测值在变 + 目标也在快速变 -> 优化目标不稳定
```

Stop-gradient 建立角色不对称，EMA 让 target 缓慢跟随 context encoder，为在线分支提供更平滑的目标。

但不要把 EMA 说成数学上单独保证永不坍塌。I-JEPA 的训练行为还依赖 predictor、架构、归一化、优化过程和 mask 设计。

---

## 9. 一次训练 step 的伪代码

```python
# x: [B, C, H, W]

context_mask, target_masks = mask_sampler(x)

# Online/context branch: receives gradients.
z_context = context_encoder(x, context_mask)

# Target branch: no gradients.
with torch.no_grad():
    z_full = target_encoder(x)
    z_target = gather_target_tokens(z_full, target_masks)

# Predictor gets context content and target positions, not target pixels.
z_pred = predictor(
    z_context=z_context,
    context_positions=context_mask,
    target_positions=target_masks,
)

assert z_pred.shape == z_target.shape
loss = torch.mean(torch.abs(z_pred - z_target))

optimizer.zero_grad()
loss.backward()
optimizer.step()

with torch.no_grad():
    for p_target, p_context in zip(
        target_encoder.parameters(),
        context_encoder.parameters(),
    ):
        p_target.data.mul_(momentum).add_(
            p_context.data,
            alpha=1.0 - momentum,
        )
```

这段伪代码省略了多 target blocks 的 batch 重排、分布式训练、混合精度、学习率和 momentum schedule。今天只检查三个不变量：

```text
1. predictor 没看到 target pixels；
2. z_pred 与 z_target 对齐；
3. target encoder 没有 optimizer gradient。
```

---

## 10. 用一个玩具图像追踪数据流

设：

```text
B = 2
image = 96 × 96 RGB
patch = 8 × 8
D_e = 192
```

则：

```text
patch grid = 12 × 12
N = 144
all patch tokens = [2, 144, 192]
```

假设 context 有 80 个 patches，一个 target block 有 16 个 patches：

```text
context encoder input:  [2, 80, 192]
z_context:              [2, 80, 192]

target encoder input:   [2, 144, 192]
full target features:   [2, 144, 192]
gather target features: [2, 16, 192]

predictor output:       [2, 16, 192]
L1 compares:            [2, 16, 192] vs [2, 16, 192]
```

若一张图使用 4 个相同大小的 target blocks，可以概念性写成：

```text
z_pred:   [2, 4, 16, 192]
z_target: [2, 4, 16, 192]
```

---

## 11. 从信息流角度理解 I-JEPA

把每个模块压缩成一个问题：

```text
Mask sampler：模型看哪里、猜哪里？
Context encoder：可见内容意味着什么？
Target encoder：正确的 latent 答案是什么？
Predictor：给定 context，在指定位置应出现什么表征？
Loss：预测答案和目标答案相差多少？
EMA：如何让答案生成器缓慢跟随学习者？
```

I-JEPA 不要求 predictor 生成唯一正确的像素排列。它要求预测 target encoder 所保留的 representation。因此训练目标是否具有语义，取决于 target encoder 表征、mask 尺度以及训练动态。

---

## 12. I-JEPA 为什么是“联合嵌入预测”

“Joint embedding”表示 context 与 target 最终在可比较的 representation space 中处理；“predictive”表示模型并非只让两个 view 静态对齐，而是利用 context 和 target position 去预测 target representation。

可以区分三种典型目标：

```text
Contrastive：让相关 view 接近、不相关样本远离。
MAE：       从可见 patches 重建被遮挡 pixels。
I-JEPA：    从 context representations 预测 target representations。
```

当你看到一个新方法时，先问：

```text
输入分支看到了什么？
预测目标位于 pixel space 还是 latent space？
目标是固定标签、另一视图，还是 EMA encoder 的输出？
哪条路径接收梯度？
是否存在信息泄漏或平凡解？
```

---

## 13. 常见误区

### 误区 1：Context encoder 看到了整张图，只是在输出处删掉 target

错误。若 target pixels 在 context encoding 时参与 self-attention，就已造成信息泄漏。Context branch 必须从输入层面排除 target 内容。

### 误区 2：Target encoder 也必须只看 target patches

错误。Target encoder 可以在完整图像上产生上下文化 features，再抽取 target positions 作为监督信号。

### 误区 3：Predictor 不应该知道 target position

错误。知道“要回答第几题”不等于知道答案。位置条件是区分不同 target blocks 所必需的。

### 误区 4：Target encoder 由 optimizer 和 EMA 同时更新

错误。标准角色划分中，它不接收 loss 梯度，由 context encoder 参数的 EMA 更新。

### 误区 5：Stop-gradient 等于把 target encoder 永久冻结

错误。它不通过 backprop 更新，但仍通过 EMA 随训练缓慢变化。

### 误区 6：`z_pred` 与 `z_target` 只要 token 数相同即可

不够。二者还必须在 batch、target block、空间位置和 feature 维度上正确对应。

### 误区 7：Loss 降低证明 representation 已经具有语义

错误。还需要 linear probe、fine-tuning、kNN、迁移任务或表征统计进行验证。

### 误区 8：I-JEPA 是一个可以直接执行动作的 world model

错误。原始 I-JEPA 学习图像内的 representation prediction，没有显式 action-conditioned dynamics。要用于控制，还需加入：

```text
z_t + action_t -> z_{t+1}
```

以及任务 cost 和 planner。

---

## 14. 与 world model 和规划的衔接

I-JEPA 已经具备 world model 的一个关键思想：**在 latent space 中预测，而不是在 pixel space 中复原所有细节。**

但它与可规划的 dynamics model 还有距离：

| I-JEPA | Action-conditioned latent world model |
|---|---|
| 同一图像内 context → target latent | 当前 latent + action → future latent |
| target position 作为条件 | action 和时间作为条件 |
| 学空间语义表征 | 学可控制的状态转移 |
| 不直接执行多步 rollout | 需要递归预测多步 future latent |

可迁移的抽象模式是：

```text
已知信息 + 条件 -> predictor -> 未知部分的 latent
```

在 I-JEPA 中：

```text
context latent + target position -> target latent
```

在控制中：

```text
current latent + action -> next latent
```

边界是：仅学会图像内遮挡预测，并不保证 latent 对动作、可达性或长期 rollout 友好，这些性质必须用额外数据和实验验证。

---

## 15. 今日理解检查

### 1. Context encoder 和 target encoder 分别看到了什么？

```text
Context encoder：只看允许的 context patches。
Target encoder：看完整图像，并从输出中抽取 target positions 的 features。
```

### 2. 为什么 target encoder 看完整图像不算 predictor 偷看答案？

因为 target branch 只用于生成 stop-gradient 监督信号；target pixels 没有作为 predictor 的输入。

### 3. Predictor 为什么需要 target position？

因为相同 context 可能对应多个待预测区域。位置条件告诉 predictor 当前输出对应哪个空间位置。

### 4. 哪些参数接收 optimizer 更新？

```text
context encoder + predictor
```

### 5. Target encoder 如何更新？

```math
\xi \leftarrow m\xi+(1-m)\theta
```

### 6. 为什么 `z_pred` 与 `z_target` 必须 shape 和位置对齐？

Loss 需要逐个 target token 比较对应表征。错位比较会让模型学习错误的空间映射。

### 7. Target block 太小可能发生什么？

任务可能主要依赖局部纹理、边缘延续等捷径，不必形成较高层语义表征。

### 8. Context 太稀疏可能发生什么？

目标可能从输入中不可预测，导致训练信号噪声大或 predictor 倾向输出平均化表示。

---

## 16. 三个练习

### 练习 1：闭卷补全数据流

填写括号：

```text
x + context mask -> (              ) -> z_context

x -> (              ) -> full features
   -> gather target positions -> stop-gradient(z_target)

z_context + (              ) -> predictor -> z_pred

loss = (              )
```

提示 1：两个 encoder 的输入可见范围不同。  
提示 2：predictor 还需要知道“预测哪里”。

答案：

```text
context encoder
target encoder
target position information
mean L1(z_pred, z_target)
```

### 练习 2：找出泄漏 bug

下面的实现有什么问题？

```python
all_features = context_encoder(full_image)
z_context = all_features[:, context_indices]
z_pred = predictor(z_context, target_positions)
```

提示：想想 self-attention 发生在筛选前还是筛选后。

答案：context encoder 已在完整 token 序列上做过 self-attention。即使输出阶段只保留 context positions，其 features 也可能已经融合 target pixels，因此发生信息泄漏。应该在进入 context encoder 前排除 target patches。

### 练习 3：迁移到 action-conditioned world model

把下列 I-JEPA 元素映射到机器人控制：

```text
context information -> ?
target condition     -> ?
target latent        -> ?
predictor            -> ?
```

一种答案：

```text
context information -> current state latent z_t
target condition     -> action a_t（以及可选的时间/horizon）
target latent        -> next state latent z_{t+1}
predictor            -> latent dynamics model
```

---

## 17. 不看笔记时应该能复述的版本

### 17.1 30 秒版本

I-JEPA 从一张图像中采样一个可见 context 和若干 target blocks。Context encoder 只编码可见 patches。Target encoder 对完整图像生成上下文化 patch features，再抽取 target positions 作为停止梯度的监督信号。Predictor 接收 context features 和 target position information，输出对应位置的预测 representation。训练用 L1 对齐 `z_pred` 和 `z_target`；context encoder 与 predictor 由 optimizer 更新，target encoder 由 context encoder 参数的 EMA 更新。

### 17.2 五行记忆版

```text
Mask 决定模型看哪里、猜哪里。
Context encoder 只编码可见 patches。
Target encoder 从完整图像产生 latent 答案。
Predictor 用 context + target position 预测 target latent。
Optimizer 更新 context/predictor，EMA 更新 target encoder。
```

### 17.3 最短主线

```text
visible context + target position -> predicted target representation
```

---

## 18. 今日完成标准

- [ ] 能画出 mask sampler、context encoder、target encoder、predictor。
- [ ] 能说清 context branch 与 target branch 分别看到了什么。
- [ ] 能解释 target encoder 为什么看完整图像却不造成输入泄漏。
- [ ] 能解释 predictor 为什么需要 target positional information。
- [ ] 能追踪 `x -> tokens -> z_context/z_target -> z_pred -> loss` 的 shape。
- [ ] 能标出 stop-gradient 的位置。
- [ ] 能写出 target encoder 的 EMA 更新公式。
- [ ] 能指出 optimizer 更新哪些模块。
- [ ] 能解释 target block 太小和 context 太稀疏的风险。
- [ ] 能判断一段伪代码是否让 context encoder 偷看 target pixels。
- [ ] 能说明 loss 下降为什么不等于 representation 一定有用。
- [ ] 能把 I-JEPA 模式迁移成 `z_t + a_t -> z_{t+1}`。

完成今天的最低门槛不是“看完本文”，而是：**关掉笔记后，独立画出四模块、两条更新路径，并正确标出 predictor 能看到位置但看不到 target pixels。**

---

## 19. 与 Day 7 的衔接：第一周复盘

明天需要把前六天内容压缩成一个可迁移的判断：

```text
为什么 JEPA 式 latent prediction 可能比 pixel prediction
更适合作为 world model 的 representation learning 起点？
```

准备从四个角度回答：

1. 预测空间：pixel 还是 latent；
2. 不确定性：是否被迫还原不可预测细节；
3. 表征质量：如何避免或检查 collapse；
4. 规划接口：如何从空间 target position 迁移到 action-conditioned dynamics。

明天不要只重复“latent 更语义”。需要补上边界：

```text
latent 是否适合 planning，必须用 dynamics rollout、goal distance、
control success rate 等实验验证，不能从预训练 loss 直接推出。
```

---

## 20. 今日学习日志

```text
日期：2026-07-14
主题：I-JEPA 架构拆解

今天学到的核心概念：
I-JEPA 用 context encoder 编码可见 patches，用 EMA target encoder 产生目标表征，
再让 predictor 根据 context features 和 target positions 预测 target representations。

我能举出的例子：
对于 12×12 patch 网格，context encoder 可只看其中 80 个 patches，
predictor 根据目标位置预测一个包含 16 个 patches 的 target block 表征。

我还不清楚的点：
1.
2.

今天的产出文件或结果：
notes/day06_ijepa_architecture.md

明天第一步：
闭卷画出 I-JEPA 训练图，然后回答“为什么 JEPA 可能更适合 world model”。
```
