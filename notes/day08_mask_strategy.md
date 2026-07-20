<!-- fullWidth: false tocVisible: false tableWrap: true -->
# Day 8：I-JEPA Mask Strategy——遮哪里，决定模型学什么

计划日期：**2026-07-16 周四**  
接续日期：**2026-07-20 周一**  
主题：**理解 target block scale、context block 与信息泄漏，并能解释 mask 为什么会影响表征的语义层次**

前置笔记：

- [Day 5：ViT 基础](day05_vit_basics.md)
- [Day 6：I-JEPA 架构拆解](day06_ijepa_architecture.md)
- [Day 7：第一周复盘](week01_review.md)

参考资料：

- I-JEPA 论文：*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*  
  <https://arxiv.org/abs/2301.08243>
- I-JEPA 官方实现：`facebookresearch/ijepa`  
  <https://github.com/facebookresearch/ijepa>
- 官方 ViT-H/14 配置：  
  <https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml>
- 官方 multi-block mask 实现：  
  <https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py>

> 今天只抓一个问题：**为什么 I-JEPA 不是“随机遮住一些 patch”这么简单，而要同时设计较大的 target blocks 与信息充分、空间分散的 context？**

---

## 0. 今天的直接答案

Mask strategy 实际上在定义模型面对的预测问题：

```text
target 太小：模型可能靠相邻纹理、边缘延续等局部捷径作答
target 足够大：模型更需要利用物体结构、场景布局等高层信息

context 太少：目标从输入中不可预测，模型只能输出平均化答案
context 太局部：模型仍可能只靠近邻纹理，不必理解全局关系
context 足够且分散：模型必须整合图像不同位置的信息
```

I-JEPA 的核心组合是：

```text
一个信息充分、空间分散的 context
        +
多个相对较大的连续 target blocks
        ↓
从可见全局结构预测不可见区域的 target representations
```

所以，mask 不是单纯提高难度。它要把任务放在一个合适区间：

```text
不能简单到可走局部捷径
也不能难到目标几乎不可预测
```

---

## 0.1 今日安排（约 90 分钟）

| 时间 | 内容 | 必须留下的结果 |
|---:|---|---|
| 0–15 分钟 | 阅读第 1–3 节 | 能区分 context mask 与 target mask |
| 15–35 分钟 | 阅读第 4–6 节 | 手算 16×16 patch grid 上的 mask 面积 |
| 35–50 分钟 | 完成第 7 节 | 解释 target 太小与 context 太少的不同后果 |
| 50–65 分钟 | 阅读第 8–10 节 | 能指出两种信息泄漏 bug |
| 65–80 分钟 | 完成第 14 节练习 | 画一张合法 mask 与一张失败 mask |
| 80–90 分钟 | 填写理解检查和日志 | 留下 Day 9 的 EMA 问题 |

如果今天只有 45 分钟：

```text
1. 读第 0、3、7、10 节；
2. 手算一次官方配置的 target patch 数；
3. 完成练习 2“判断 mask 是否合理”；
4. 填写今日学习日志。
```

---

## 1. Mask strategy 在 I-JEPA 中处于哪里

先回忆完整数据流：

```text
                              image x
                         ┌────────┴────────┐
                         │                 │
                  context indices     full image
                         │                 │
                         v                 v
                 context encoder     target encoder
                         │                 │
                         │          gather target indices
                         │                 │
                         v                 v
                    z_context          z_target
                         │                 │
              + target positions          │ stop-gradient
                         │                 │
                         v                 │
                     predictor            │
                         │                 │
                         v                 │
                       z_pred ─── L1 ──────┘
```

Mask sampler 至少决定三件事：

1. `context encoder` 能看到哪些 patch；
2. `predictor` 需要回答哪些 target positions；
3. 一次训练中预测区域的大小、形状、位置和数量。

它没有直接改变 loss 公式，却改变了 loss 所对应的问题。

---

## 2. 三个容易混淆的对象

### 2.1 Target block

Target block 是 patch 网格上的一个连续矩形区域。Target encoder 对完整图像编码后，从这些位置抽取 representations，作为预测目标。

```text
target block 决定：模型要猜哪里
```

### 2.2 Context block

Context block 是交给 context encoder 的可见 patch 集合。官方 multi-block 实现先采样一个较大的 encoder rectangle，再排除不允许与 target 重叠的位置，因此最终 context 往往像一个带有若干孔洞的分散区域。

```text
context block 决定：模型可以依据什么信息作答
```

### 2.3 Target position token

Predictor 可以知道 target 在哪里，但不能知道 target pixels 是什么。

```text
位置条件：告诉模型“回答哪一题”
target pixels：直接把“答案内容”交给模型
```

二者不能混为一谈。

---

## 3. 官方配置中的关键数字

I-JEPA 官方 ImageNet ViT-H/14、224×224 配置使用：

```yaml
patch_size: 14
num_pred_masks: 4
pred_mask_scale: [0.15, 0.20]
enc_mask_scale: [0.85, 1.00]
aspect_ratio: [0.75, 1.50]
allow_overlap: false
```

对应 patch grid：

```text
224 / 14 = 16
grid = 16 × 16
total patches = 256
```

单个 target block 的名义面积约为全部 patches 的 15%–20%：

```text
256 × 0.15 = 38.4
256 × 0.20 = 51.2
```

因此单个 target block 大约覆盖 38–51 个 patch，实际数量会受到整数取整、长宽比和边界限制影响。

一次采样 4 个 target blocks。不要直接计算：

```text
4 × 20% = 80%，所以一定遮住 80%
```

因为 target blocks 之间可能重叠；`allow_overlap: false` 在该实现中主要限制 encoder/context mask 与 predictor/target masks 的重叠，不代表四个 target blocks 必然互不重叠。

---

## 4. Scale 到底表示什么

设 patch grid 高宽为 `H_p × W_p`，总 patch 数为：

```math
N=H_pW_p
```

采样一个 scale `s` 后，代码先计算名义面积：

```math
A=\lfloor Ns\rfloor
```

再结合长宽比 `r` 得到矩形高宽：

```math
h\approx\sqrt{Ar},\qquad w\approx\sqrt{A/r}
```

最后把 `h`、`w` 取整并限制在 patch grid 内。

例如 `N=256`、`s=0.18`、`r=1`：

```text
A = floor(256 × 0.18) = 46
h ≈ sqrt(46) ≈ 6.8
w ≈ sqrt(46) ≈ 6.8

取整后可得到约 7 × 7 = 49 patches
```

所以 scale 是采样矩形的面积比例参数，不是最终有效 token 数的绝对保证。

---

## 5. 为什么 target block 需要足够大

### 5.1 小 target 容易出现局部捷径

假设只遮住一个 patch：

```text
[砖墙][砖墙][ ? ][砖墙]
```

模型可能只需根据左右纹理延续预测目标，不必理解这是建筑、街道还是室内场景。

类似捷径包括：

- 边缘方向连续；
- 颜色与纹理平滑；
- 相邻 patch 高度相关；
- 局部重复图案。

这些能力不是完全无用，但它们不足以迫使 encoder 形成高层语义。

### 5.2 较大 target 提高语义需求

如果 target 覆盖狗的头部、汽车前半部分或人物上身，模型不能仅复制一个邻近 patch。它需要利用：

- 物体整体形状；
- 部件之间的关系；
- 场景中物体常出现的位置；
- context 所暗示的姿态与朝向。

这就是论文所说的“足够大的 target 倾向于语义化”的直觉。

### 5.3 大并不是越大越好

若 target 覆盖几乎整张图，而 context 只剩很少信息：

```text
目标不再由 context 充分决定
    -> 多种内容都可能正确
    -> 确定性 predictor 容易输出平均 representation
    -> 训练变得噪声大或不稳定
```

因此目标是“足够大”，不是“最大”。

---

## 6. 为什么 context 要信息充分且空间分散

### 6.1 只看局部 context 的问题

若 context 只位于 target 左侧的一小块区域，模型仍可能靠局部纹理外推：

```text
[context][target]
```

它不必整合图像远处的信息。

### 6.2 分散 context 的价值

如果可见 patch 分布在 target 周围甚至图像多个位置：

```text
context ─┐
context ─┼─> 判断场景与物体结构 ─> target latent
context ─┘
```

predictor 更可能需要使用长距离关系：

- 上方的天空与下方的道路共同提示街景；
- 身体、四肢与背景共同提示被遮挡的头部；
- 房屋轮廓的两端共同约束中间缺失区域。

### 6.3 Context 也不能太少

Context 数量不足时，问题可能从“困难但可预测”变成“信息论上含糊”。

```text
困难：需要整合多个线索才能作答
含糊：输入中根本没有足够线索区分多个答案
```

好的 mask strategy 追求前者，避免后者。

---

## 7. 四种 mask 组合的后果

| Target | Context | 模型可能学到什么 | 主要风险 |
|---|---|---|---|
| 小 | 局部 | 纹理与边缘插值 | 局部捷径 |
| 大 | 很少 | 模糊先验或平均答案 | 目标不可预测 |
| 小 | 分散且充分 | 全局信息可能被使用，但没有必要 | 任务仍过简单 |
| 较大 | 分散且充分 | 物体结构与场景语义 | I-JEPA 期望区间 |

记忆规则：

```text
target scale 控制“答案需要多高层”
context quality 控制“答案是否有依据”
```

这是直觉性描述，不是严格的一一对应定律。最终仍需用消融实验与下游评估验证。

---

## 8. 为什么使用 block masking，而不是独立随机 patch masking

独立随机 mask 可能形成椒盐状分布：

```text
可见 遮挡 可见 遮挡 可见
遮挡 可见 遮挡 可见 遮挡
```

每个遮挡 patch 周围往往都有大量可见邻居，局部插值非常方便。

连续 block masking 会制造真正缺失的空间区域：

```text
可见 可见 可见 可见
可见 [ 遮挡 遮挡 ] 可见
可见 [ 遮挡 遮挡 ] 可见
可见 可见 可见 可见
```

这增加了使用长距离信息的必要性，也更接近“从部分观察推断未知区域”的问题。

---

## 9. 多个 target blocks 的作用

一次从同一 context 预测多个 target blocks，有三个直接作用：

1. 同一张图提供多个空间预测任务；
2. predictor 必须根据不同 target positions 输出不同答案；
3. 训练信号覆盖图像多个区域，提高样本利用率。

抽象写法：

```math
\hat z_{T_k}=g_\phi(z_C, pos_{T_k}),\qquad k=1,\ldots,K
```

```math
\mathcal L=\frac{1}{K}\sum_{k=1}^{K}
\operatorname{L1}(\hat z_{T_k},\operatorname{sg}(z_{T_k}))
```

多个 blocks 不表示 predictor 同时看到这些 target 的内容。它只接收各 target 的位置条件。

---

## 10. 信息泄漏：Mask 实现最危险的 bug

### 10.1 Bug A：编码完整图像后再筛 context tokens

错误实现：

```python
all_features = context_encoder(full_image)
z_context = all_features[:, context_indices]
```

虽然最后只保留 context positions，但 self-attention 已在完整 token 序列上发生。保留下来的 features 可能已经融合 target pixels。

正确不变量：

```text
target pixels 必须在进入 context encoder 之前被排除
```

### 10.2 Bug B：把 target patch embedding 作为 predictor 输入

错误实现：

```python
z_pred = predictor(z_context, target_patch_embeddings)
```

这相当于把 target 内容直接交给 predictor。

合法输入应是：

```python
z_pred = predictor(z_context, target_position_embeddings)
```

### 10.3 Bug C：Context 与 target indices 意外重叠

若同一 patch 同时出现在 context 和 target 中，模型可以直接复制该位置的信息。实现中必须明确检查：

```python
assert set(context_indices).isdisjoint(set(target_indices))
```

如果设计允许某种 overlap，也必须清楚它改变了学习问题，不能把它当作标准 I-JEPA 的无泄漏设置。

---

## 11. Shape tracing：16×16 patch grid 示例

设：

```text
B = 32
image = 224 × 224
patch = 14 × 14
grid = 16 × 16
N = 256
D = 1280
K = 4 target blocks
```

若一次采样后，每个样本统一保留 180 个 context patches、每个 target block 保留 48 个 target patches：

```text
context indices: [B, 180]
target indices:  [K, B, 48] 或等价重排

z_context: [B, 180, D]
z_target:  [K, B, 48, D]
z_pred:    [K, B, 48, D]
```

实现可能把 `K` 合并进 batch：

```text
z_target: [K×B, 48, D]
z_pred:   [K×B, 48, D]
```

关键不变量不是轴的写法，而是：

```text
同一个样本
+ 同一个 target block
+ 同一个空间位置
必须在 z_pred 与 z_target 中严格对应
```

---

## 12. Mask 策略也是一种 inductive bias

Mask sampler 没有人工类别标签，却把一种偏好写进训练任务：

```text
图像中相隔较远的区域之间存在可预测的结构关系
```

这适合自然图像，因为物体与场景通常具有空间一致性。但迁移到其他数据时，不能机械照搬二维矩形 mask。

### 视频

```text
空间 block -> 时空 tube
需要同时考虑空间范围与时间跨度
```

### 无线信道图/CKM

```text
二维图像区域 -> 地理区域、频段、时间窗口或 Tx-Rx 条件块
```

### 机器人轨迹

```text
空间 patch -> 状态片段、观测片段或未来时间段
```

迁移规则不是“永远遮矩形”，而是：

> 遮掉一个具有结构意义、不能靠邻近复制解决、但仍能由剩余信息推断的目标块。

---

## 13. 如何为 mini-I-JEPA 设计第一版 mask

不要一开始复刻大规模 ViT-H 配置。对 CIFAR-10 或 STL-10 的小模型，先建立可调但简单的版本：

```text
1. 把图像变成 H_p × W_p patch grid；
2. 采样 2–4 个连续 target rectangles；
3. 单个 target 面积先试 15%–20%；
4. context 使用 target 之外的大部分 patches；
5. 明确保证 context-target 不重叠；
6. 记录实际 context ratio、target union ratio 与重叠率；
7. 保存 mask 可视化，而不是只打印 indices。
```

建议至少比较两个设置：

| 设置 | Target | Context | 预期 |
|---|---|---|---|
| A：局部捷径 | 小随机 patches | 大量邻近可见 patches | loss 可能易降，语义较弱 |
| B：block prediction | 15%–20% 连续 blocks | 充分且分散 | 任务更难，可能更利于语义 |

不要只比较预训练 loss。还应比较：

- linear probe accuracy；
- feature standard deviation / effective rank；
- nearest-neighbor 语义一致性；
- 收敛速度和训练稳定性。

---

## 14. 三个立即练习

### 练习 1：手算 mask

一个 `12×12` patch grid 共有多少 patches？单个 target scale 为 `0.20` 时，名义 target 面积约是多少？

提示 1：先计算 `12×12`。  
提示 2：再乘 `0.20`。  

<details>
<summary>答案</summary>

```text
N = 12 × 12 = 144
144 × 0.20 = 28.8
```

名义面积约 28–29 个 patches；实际矩形面积还会受到长宽比与取整影响。

</details>

### 练习 2：判断哪一个更合理

```text
方案 A：遮住 1 个 patch，周围 8 个 patch 全部可见
方案 B：遮住一个约占图像 18% 的连续区域，context 分布在其四周
方案 C：遮住 90% 图像，只留下两个无关 patch
```

先分别判断它们的主要风险。

<details>
<summary>答案</summary>

- A：容易依赖局部纹理或边缘插值；
- B：更接近 I-JEPA 希望的困难但可预测区间；
- C：context 信息不足，target 高度含糊。

</details>

### 练习 3：找泄漏

```python
tokens = patch_embed(x)
features = context_encoder(tokens)
z_context = gather(features, context_indices)
```

问题在哪里？

<details>
<summary>答案</summary>

`context_encoder` 在 gather 之前处理了所有 tokens，target token 已参与 self-attention。应在进入 context encoder 前先选择 context tokens。

</details>

---

## 15. 今日理解检查

不看上文回答：

1. 为什么 target block 太小会鼓励局部捷径？
2. 为什么 target block 不是越大越好？
3. “context 信息充分”和“context 空间分散”分别解决什么问题？
4. Predictor 知道 target position 为什么不算泄漏？
5. 为什么编码完整图像后再筛 context features 已经太晚？
6. `pred_mask_scale=0.2` 是否表示最终一定恰好保留 20% target tokens？
7. 多个 target blocks 的总覆盖率为什么不能直接相加？
8. 如果预训练 loss 更低，能否断言 mask strategy 更好？

完成标准：

- [ ] 能画出 context、target 与 predictor position token 的关系；
- [ ] 能解释“大 target + 分散 context”的设计动机；
- [ ] 能手算给定 patch grid 和 scale 的名义面积；
- [ ] 能指出 context-target overlap 与先编码后筛选两种泄漏；
- [ ] 能设计一组 mask scale 消融，并选择 linear probe 作为表征评估之一；
- [ ] 能把 block masking 原则迁移到视频、轨迹或 CKM。

---

## 16. 不看笔记时应该能复述的版本

### 30 秒版本

I-JEPA 的 mask strategy 不是随机删 patch，而是在定义预测任务。较大的连续 target blocks 降低局部纹理插值等捷径，促使模型使用物体与场景级信息；信息充分且空间分散的 context 则保证目标仍有依据，并鼓励长距离信息整合。Predictor 可以知道 target 位置，但 context encoder 在输入阶段就必须看不到 target pixels。Mask 太简单会学捷径，太难会让目标不可预测，所以需要通过 mask 消融和 linear probe 等下游指标验证。

### 五行记忆版

```text
Mask 决定模型看哪里、猜哪里。
小 target 容易靠局部纹理作答。
大 target 更需要物体与场景结构。
Context 要充分且分散，但不能包含 target pixels。
最佳 mask 看下游表征，不只看预训练 loss。
```

### 最短主线

```text
遮挡方式 -> 预测难度与可用线索 -> 模型被迫学习的表征层次
```

---

## 17. 与 Day 9 的衔接：Target encoder 为什么用 EMA

Mask strategy 已经定义了“看哪里、猜哪里”，但 target representations 本身也在训练过程中变化。

下一步要回答：

```text
为什么 target encoder 不通过当前 loss 直接更新？
为什么用 ξ <- mξ + (1-m)θ？
momentum m 越接近 1，target 变化会怎样？
EMA 是稳定目标，还是保证绝不坍塌的定理？
```

---

## 18. 今日学习日志

```text
计划日期：2026-07-16
实际接续日期：2026-07-20
主题：I-JEPA mask strategy

今天学到的核心概念：

我能举出的例子：

Target 太小时的风险：

Context 太少时的风险：

我能指出的信息泄漏方式：
1.
2.

我还不清楚的点：
1.
2.

今天的产出文件：
notes/day08_mask_strategy.md

下一步：
学习 EMA target encoder，手写 ξ <- mξ + (1-m)θ 并解释每一项。
```
