<!-- fullWidth: false tocVisible: false tableWrap: true -->
# Day 1: I-JEPA 必读部分提取笔记

论文: **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**  
作者: Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas  
链接: <https://arxiv.org/abs/2301.08243>

> 说明: 这份笔记不是逐字搬运论文，而是把今天必须读的 Abstract、Introduction、Method、核心图表重写成可直接学习的 Markdown。图为论文原图截图，表格为关键结果转写和压缩。

---

## 0. 今天只抓一个问题

**I-JEPA 到底预测什么？**

答案: **不是预测像素，而是预测被遮挡图像区域的 latent representation。**

更具体地说:

```text
给定同一张图像中的 context block
模型要预测若干 target blocks 的 target-encoder 表征
loss 施加在 representation space
而不是 pixel space
```

这就是它和 MAE 的核心差别:

| 方法 | 输入缺失后要恢复什么 | 训练目标在哪个空间 | 学到的倾向 |
|---|---|---|---|
| MAE / 生成式 masked modeling | 像素或视觉 token | pixel/token space | 细节、纹理、局部重建能力强 |
| DINO / SimCLR / BYOL | 不同增强视图的一致表征 | embedding space | 语义强，但依赖手工增强 |
| I-JEPA | target block 的表征 | learned representation space | 不靠手工增强，逼模型预测语义级缺失信息 |

---

## 1. Abstract 提取

I-JEPA 的目标是做图像自监督表征学习，但它不走两条老路:

1. 不像 SimCLR/DINO 那样依赖大量手工图像增强来制造不同 view。
2. 不像 MAE 那样把缺失区域重建成像素。

它做的是:

```text
从一张图像中取一个 context block
让模型预测同一张图像中多个 target blocks 的表征
```

论文认为 I-JEPA 成功的关键不是某个复杂 loss，而是 **masking strategy**:

| 设计点 | 为什么重要 |
|---|---|
| target block 要足够大 | 小 patch 容易变成纹理补全；大区域更像语义预测 |
| context block 要信息充足且空间分散 | context 太少会让任务不可能；context 太完整又会让任务太简单 |
| 预测 representation 而不是 pixel | 避免模型把能力浪费在低层细节重建上 |

一句话:

> I-JEPA 用局部上下文预测其他区域的抽象表征，从而学习语义图像表示。

---

## 2. Introduction 提取: 作者为什么要提出 I-JEPA

### 2.1 现有路线 A: invariance-based SSL

代表方法:

- SimCLR
- BYOL
- DINO
- iBOT
- MSN

基本思想:

```text
同一张图像经过不同增强
模型应该输出相似 embedding
```

优点:

- 线性评估通常很强。
- 表征比较语义化。

问题:

- 依赖人为设计的数据增强，比如 crop、color jitter、blur。
- 不同任务需要的 invariance 不一样。
- 图像增强技巧不一定能迁移到视频、音频、机器人轨迹、无线信道等模态。

对你后面做 world model 的启发:

```text
如果系统未来要预测状态变化、动作后果、环境结构，
那就不能只靠“两个增强视图保持一致”来定义学习目标。
```

### 2.2 现有路线 B: generative masked modeling

代表方法:

- MAE
- BEiT
- SimMIM

基本思想:

```text
遮掉一部分输入
再把缺失部分重建回来
```

优点:

- 不太依赖手工增强。
- 可迁移到多模态。
- 训练任务直观。

问题:

- 如果重建目标是像素，模型可能过度关注纹理、颜色、边缘等低层细节。
- 语义分类这类任务上，off-the-shelf 表征往往不如 invariance-based 方法。
- 经常需要较强的 fine-tuning 才能把表征用好。

### 2.3 I-JEPA 的定位

I-JEPA 想做一个折中:

```text
像 MAE 一样，不依赖手工增强
像 DINO 一样，在表征空间学语义
但不需要负样本或复杂 view 设计
```

核心假设:

```text
如果 target representation 已经去掉了一些像素级噪声，
那么预测它会迫使 context encoder 学更高层的语义结构。
```

---

## 3. Figure 2 原图: 三类自监督架构

论文 Figure 2 对比三种架构。下面插入的是 arXiv 源文件中的原始子图。

### 3.1 Joint-Embedding Architecture

![Figure 2a: Joint-Embedding Architecture](assets/figure2a_joint_embedding.png)

理解:

```text
把同一图像的不同增强视图拉近。
关键在于 view augmentation 设计。
```

典型问题:

```text
如果没有额外机制，所有输入都输出同一个 embedding 会造成 collapse。
```

### 3.2 Generative Architecture

![Figure 2b: Generative Architecture](assets/figure2b_generative.png)

这张图可以按下面的顺序读:

```text
x -> x-encoder -> decoder -> y_hat
y ---------------------------> D(y_hat, y)
z -> decoder
```

含义:

```text
x 是已知条件。
y 是要预测/重建的真实目标。
decoder 根据 x 的表示和额外变量 z 生成 y_hat。
loss D(y_hat, y) 直接比较生成结果和真实目标。
```

在 MAE 里可以理解成:

```text
x = 没被遮住的 image patches
y = 被遮住的真实 pixels/patches
y_hat = decoder 重建出来的 pixels/patches
```

主要问题:

```text
如果目标是 pixel，模型会学习很多低层细节；
这些细节不一定等价于语义理解。
```

### 3.3 Joint-Embedding Predictive Architecture

![Figure 2c: Joint-Embedding Predictive Architecture](assets/figure2c_jepa.png)

这张图和 generative 图最关键的差异是:

```text
Generative 比较 D(y_hat, y)
JEPA 比较 D(s_hat_y, s_y)
```

也就是说:

```text
不是恢复 y 本身，而是恢复 y 的 embedding。
I-JEPA 就是这种结构在图像上的实例。
```

---

## 4. Method 提取: I-JEPA 是怎么训练的

### 4.1 总体流程

```text
输入图像
-> 切成 ViT patch tokens
-> target encoder 处理完整图像，得到每个 patch 的 target representation
-> 从 target representation 中采样多个 target blocks
-> 从原图中采样 context block，并移除与 target 重叠的 patch
-> context encoder 只处理 visible context patches
-> predictor 根据 context representation + target position tokens 预测 target representations
-> 用 representation-space distance 训练 context encoder 和 predictor
-> target encoder 用 EMA 更新，不直接反传
```

### 4.2 Figure 3 原图: I-JEPA 训练结构

![Figure 3: I-JEPA training structure](assets/figure3_ijepa.png)

最关键的一句:

```text
target encoder 看完整图像，用来产生比较语义化的 target representation；
context encoder 只能看 context，要学会预测 target representation。
```

---

## 5. Figure 4 原图: multi-block masking

论文中的 masking 不是随机丢一堆 patch，而是采样几个较大的 target blocks，再采样一个较大的 context block。

![Figure 4: Multi-block masking examples](assets/figure4_masking.png)

### 5.1 ASCII 辅助理解图

```text
原图 patch grid:

[ ][ ][ ][ ][ ][ ][ ][ ]
[ ][T][T][ ][ ][ ][T][T]
[ ][T][T][ ][C][C][T][T]
[ ][ ][ ][ ][C][C][ ][ ]
[ ][ ][C][C][C][C][ ][ ]
[ ][ ][C][C][ ][ ][ ][ ]
[ ][T][T][ ][ ][T][T][ ]
[ ][T][T][ ][ ][T][T][ ]

T = target block: 要预测其 representation
C = context block: context encoder 可见的区域
空白 = 不参与该次预测或被移除
```

注意:

```text
target block 可以和 context 原始采样区域重叠，
但真正送入 context encoder 前，重叠部分会被移除。
```

### 5.2 默认 masking 设置

| 组件 | 论文中的设计 | 直觉 |
|---|---|---|
| target block 数量 | 通常 4 个 | 多个目标迫使模型学更稳定的区域间语义关系 |
| target scale | 约 15%-20% 图像面积 | 目标足够大，避免退化成纹理补丁预测 |
| target aspect ratio | 约 0.75-1.5 | 允许目标块形状有变化 |
| context scale | 约 85%-100% 图像面积，再移除 target overlap | context 信息充分，但不能直接看到 target |
| context aspect ratio | 接近 1 | 保持较完整的空间上下文 |

一句话:

```text
I-JEPA 不是让模型猜一个小洞，而是让模型根据大范围上下文猜多个语义区域的表征。
```

---

## 6. Loss 和 EMA

### 6.1 Loss 的学习版公式

设:

- `z_t`: target encoder 给出的 target block representation
- `z_hat_t`: predictor 根据 context 预测出的 target representation
- `M`: target block 数量

训练目标可以理解成:

```math
L = (1 / M) * sum_{m=1}^{M} distance(z_hat_t_m, z_t_m)
```

论文中强调的是:

```text
distance 发生在 representation space，
不是 pixel space。
```

### 6.2 为什么 target encoder 不直接反传

target encoder 是 teacher，context encoder 是 student。

```text
student: context encoder + predictor
teacher: target encoder
```

更新方式:

```text
student 通过 loss 反向传播更新
teacher 用 student encoder 的 EMA 平滑更新
```

学习版公式:

```math
teacher = m * teacher + (1 - m) * student
```

直觉:

```text
如果 target encoder 也直接跟着 predictor 一起剧烈更新，
预测目标本身会不稳定。
EMA teacher 提供一个更平滑、更稳定的目标表征。
```

---

## 7. 核心表格提取

### Table 1: ImageNet linear evaluation 关键结果

这张表回答:

```text
I-JEPA 只预测 representation，不用手工增强，效果到底行不行？
```

| 方法 | 架构 | 预训练 epochs | ImageNet linear Top-1 |
|---|---:|---:|---:|
| MAE | ViT-B/16 | 1600 | 68.0 |
| MAE | ViT-L/16 | 1600 | 76.0 |
| MAE | ViT-H/14 | 1600 | 77.2 |
| CAE | ViT-L/16 | 1600 | 78.1 |
| I-JEPA | ViT-B/16 | 600 | 72.9 |
| I-JEPA | ViT-L/16 | 600 | 77.5 |
| I-JEPA | ViT-H/14 | 300 | 79.3 |
| DINO | ViT-B/8 | 300 | 80.1 |
| iBOT | ViT-L/16 | 250 | 81.0 |

今天你只需要读出两个结论:

1. 和 MAE 相比，I-JEPA 在 linear evaluation 上更强，而且训练 epochs 更少。
2. 和 DINO/iBOT 相比，I-JEPA 不需要手工 view augmentation，但已经接近强语义表征方法。

### Table 6: masking strategy ablation

这张表回答:

```text
为什么一定要 multi-block masking？
```

实验设置: ViT-B/16，I-JEPA 预训练 300 epochs，ImageNet-1% linear evaluation。

| target masking | target 数量 | context 类型 | context 平均比例 | Top-1 |
|---|---:|---|---:|---:|
| multi-block | 4 | block complement | 0.25 | 54.2 |
| rasterized quadrant | 3 | complement | 0.25 | 15.5 |
| single block | 1 | complement | 0.40 | 20.2 |
| random patches | 1 | complement | 0.40 | 17.6 |

结论:

```text
不是所有 mask 都能学到好语义。
多个较大 target blocks + 信息充分的 context，是 I-JEPA 的关键。
```

### Table 7: target 是 representation 还是 pixel

这张表是今天最重要的实验表。

| 预测目标 | 架构 | epochs | ImageNet-1% Top-1 |
|---|---:|---:|---:|
| target encoder output | ViT-L/16 | 500 | 66.9 |
| pixels | ViT-L/16 | 800 | 40.7 |

你要记住:

```text
同样是预测缺失区域，
预测 latent representation 明显优于预测 pixels。
```

这正好支撑 I-JEPA 的核心观点:

```text
representation-space prediction 能让模型少纠缠像素细节，
更多学习语义结构。
```

### Appendix Tables 8-11: masking 细节对结果的影响

这些表今天不用细读，但结论很关键。

#### target block size

| target scale | target 数量 | context scale | Top-1 |
|---|---:|---|---:|
| 0.075-0.20 | 4 | 0.85-1.00 | 19.2 |
| 0.10-0.20 | 4 | 0.85-1.00 | 39.2 |
| 0.125-0.20 | 4 | 0.85-1.00 | 42.4 |
| 0.15-0.20 | 4 | 0.85-1.00 | 54.2 |
| 0.20-0.25 | 4 | 0.85-1.00 | 38.9 |
| 0.20-0.30 | 4 | 0.85-1.00 | 33.6 |

结论:

```text
target 太小不语义，target 太大又可能太难。
0.15-0.20 这个范围表现最好。
```

#### context size

| target scale | target 数量 | context scale | Top-1 |
|---|---:|---|---:|
| 0.15-0.20 | 4 | 0.40-1.00 | 31.2 |
| 0.15-0.20 | 4 | 0.65-1.00 | 47.1 |
| 0.15-0.20 | 4 | 0.75-1.00 | 49.3 |
| 0.15-0.20 | 4 | 0.85-1.00 | 54.2 |

结论:

```text
context 不能太少。
I-JEPA 要做的是“有依据的语义预测”，不是盲猜。
```

#### number of target blocks

| target scale | target 数量 | context scale | Top-1 |
|---|---:|---|---:|
| 0.15-0.20 | 1 | 0.85-1.00 | 9.0 |
| 0.15-0.20 | 2 | 0.85-1.00 | 22.0 |
| 0.15-0.20 | 3 | 0.85-1.00 | 48.5 |
| 0.15-0.20 | 4 | 0.85-1.00 | 54.2 |

结论:

```text
多个 target blocks 会显著增强学习信号。
只预测一个 target block 效果很差。
```

#### target masking at output vs input

| target masking 位置 | 架构 | epochs | Top-1 |
|---|---:|---:|---:|
| target encoder output | ViT-H/16 | 300 | 67.3 |
| target encoder input | ViT-H/16 | 300 | 56.1 |

结论:

```text
先用 target encoder 看完整图像，再从输出表征中取 target block，更有利于形成语义目标。
如果先遮输入再让 target encoder 编码，target 表征本身会弱一些。
```

---

## 8. 五个问题的直接答案

### 1. JEPA 的 context 是什么？

在 I-JEPA 中，context 是同一张图像中被 context encoder 可见的一组 patch。它通常来自一个较大的 context block，但会移除与 target blocks 重叠的区域。

学习版理解:

```text
context = 模型已知的图像证据
```

### 2. target 是什么？

target 不是原始像素块，而是 target encoder 输出的 patch-level representation 中被采样出来的若干 blocks。

学习版理解:

```text
target = 需要被预测的语义表征区域
```

### 3. predictor 预测的是像素还是表征？

预测的是表征。

```text
predictor(context representation, target position tokens)
-> predicted target representation
```

### 4. target encoder 为什么不直接反传？

因为 target encoder 提供训练目标。如果它也跟着 loss 快速更新，目标会不稳定。I-JEPA 用 EMA 让 target encoder 成为一个平滑的 teacher。

### 5. 为什么这种方式可能比 MAE 更适合 world model？

world model 关心的是:

```text
状态结构
动作后果
长期可预测性
语义和物理关系
```

而不是:

```text
每个像素的颜色值是否恢复得完全一致
```

I-JEPA 的 latent prediction 更接近 world model 需要的抽象状态预测:

```text
o_t -> z_t
z_context -> z_target
未来可以扩展为:
z_t, a_t -> z_{t+1}
```

---

## 9. 今天的最小输出

请用自己的话写出下面 5 句话，不要查论文:

```text
1. I-JEPA 的 context 是模型能看到的图像 patch 区域。
2. I-JEPA 的 target 是 target encoder 输出的 latent representation block，不是像素。
3. predictor 根据 context representation 和 target 位置 token 预测 target representation。
4. target encoder 用 EMA 更新，是为了提供稳定的 teacher target。
5. I-JEPA 比 MAE 更接近 world model，因为它训练模型预测抽象状态，而不是重建底层像素。
```

如果你能不看笔记说清这 5 句话，Day 1 就过关。

---

## 10. 今天暂时不要读的部分

以下内容以后再看:

| 内容 | 什么时候读 |
|---|---|
| 完整实验细节 | 第 2 周以后 |
| predictor depth / width 消融 | 写 mini-I-JEPA 设计时 |
| object counting / depth prediction transfer | 做综述或汇报时 |
| Appendix 训练超参数 | 准备复现代码时 |
| V-JEPA / V-JEPA 2 | 先完成 I-JEPA 直觉和 mini-I-JEPA |

---

## 11. 今日日志模板

```text
今天学到的核心概念:
我能举出的例子:
我还不清楚的点:
今天的产出文件或结果: notes/day01_what_is_JEPA.md
明天第一步: 对比 SimCLR / MAE / I-JEPA 的输入、目标、loss
```
