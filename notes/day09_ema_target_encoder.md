<!-- fullWidth: false tocVisible: false tableWrap: true -->
# Day 9：EMA Target Encoder——为什么目标网络要慢半拍

计划日期：**2026-07-17 周五**  
接续日期：**2026-07-20 周一**  
主题：**理解 EMA target encoder 的更新公式、时间尺度、训练顺序与实现不变量**

前置笔记：

- [Day 4：表征坍塌](day04_representation_collapse.md)
- [Day 6：I-JEPA 架构拆解](day06_ijepa_architecture.md)
- [Day 8：Mask Strategy](day08_mask_strategy.md)

参考资料：

- I-JEPA 论文：*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*  
  <https://arxiv.org/abs/2301.08243>
- I-JEPA 官方训练代码：  
  <https://github.com/facebookresearch/ijepa/blob/main/src/train.py>
- I-JEPA 官方 ViT-H/14 配置：  
  <https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml>
- BYOL 论文：*Bootstrap Your Own Latent*  
  <https://arxiv.org/abs/2006.07733>

> 今天只抓一个问题：**为什么 I-JEPA 不让 target encoder 直接接收梯度，而是让它通过 context encoder 参数的指数移动平均缓慢更新？**

---

## 0. 今天的直接答案

I-JEPA 同时维护两个结构相同但更新方式不同的 encoder：

```text
context encoder θ：由当前 loss 的梯度快速更新
target encoder  ξ：不接收梯度，通过 θ 的 EMA 缓慢更新
```

EMA 更新公式是：

```math
\xi_t=m_t\xi_{t-1}+(1-m_t)\theta_t
```

其中：

- `θ_t`：当前 optimizer step 后的 context encoder 参数；
- `ξ_{t-1}`：上一步 target encoder 参数；
- `m_t`：momentum，通常非常接近 1；
- `ξ_t`：更新后的 target encoder 参数。

直觉上，target encoder 是 context encoder 最近一段训练历史的平滑版本：

```text
student/context：快速学习，允许被当前 batch 推动
teacher/target：慢速跟随，提供较稳定的 latent targets
```

EMA 的主要价值是稳定目标、建立快慢时间尺度和打破双分支的更新对称性。它不是“永不坍塌”的数学保证，也不是一个独立预训练好的老师。

---

## 0.1 今日安排（约 90 分钟）

| 时间 | 内容 | 必须留下的结果 |
|---:|---|---|
| 0–15 分钟 | 阅读第 1–3 节 | 能画出 gradient 与 EMA 两条更新路径 |
| 15–30 分钟 | 阅读第 4–5 节 | 手算一次 `m=0.9` 的 EMA 更新 |
| 30–50 分钟 | 阅读第 6–7 节 | 能解释 momentum 对时间尺度的影响 |
| 50–65 分钟 | 阅读第 8–10 节 | 能写出正确训练 step 顺序 |
| 65–80 分钟 | 完成第 14 节练习 | 找出三个 EMA 实现 bug |
| 80–90 分钟 | 完成理解检查与日志 | 留下 Day 10 的 predictor 问题 |

如果今天只有 45 分钟：

```text
1. 读第 0、2、5、8 节；
2. 手算 m=0.9 的三步 EMA；
3. 默写 target.requires_grad=False；
4. 完成练习 2“判断更新顺序”。
```

---

## 1. 从 Day 8 接上：Mask 定义问题，Target encoder 定义答案

Day 8 已经确定：

```text
context mask：模型能看什么
target mask：模型要猜什么位置
```

但 latent prediction 还需要回答：目标位置上的“正确 latent”由谁产生？

I-JEPA 的答案是：

```text
完整图像 -> target encoder ξ -> target features
                              -> 抽取 target positions
                              -> stop-gradient(z_target)
```

这意味着训练标签不是固定人工标签，而是由一个正在变化的神经网络生成。于是核心困难变成：

> 如果答案生成器也随当前 loss 快速变化，预测器到底在追什么？

EMA target encoder 就是为这个动态目标提供稳定时间尺度。

---

## 2. 两条更新路径必须分开

设：

- context encoder 参数：`θ`；
- predictor 参数：`φ`；
- target encoder 参数：`ξ`。

训练图：

```text
                         loss
                        /    \
                       /      X  stop-gradient
                      v        \
                predictor φ    z_target
                      │             │
                      v             │
             context encoder θ  target encoder ξ
                      │             ▲
                      │             │
            optimizer/gradient      │ EMA from θ
                      │             │
                      └─────────────┘
```

参数更新规则：

```text
θ <- AdamW(θ, grad_θ)
φ <- AdamW(φ, grad_φ)
ξ <- mξ + (1-m)θ
```

最重要的不变量：

```text
target encoder 不在 optimizer 参数列表中
target encoder parameters 的 requires_grad=False
target forward 不建立梯度图
target encoder 只通过 EMA 更新
```

---

## 3. Stop-gradient、冻结与 EMA 的区别

### 3.1 Stop-gradient

Stop-gradient 描述当前 forward/backward：

```text
forward：正常计算 z_target
backward：梯度不沿 target branch 传播
```

### 3.2 永久冻结

永久冻结表示 target encoder 从初始化后再也不变化：

```text
ξ_t = ξ_0
```

若它只是随机初始化网络，context encoder 将长期追逐随机且固定的特征目标，这通常不是 I-JEPA 想要的学习过程。

### 3.3 EMA 更新

EMA target 在当前 step 不接收梯度，但会在 step 结束后缓慢吸收 context encoder 的参数：

```text
无 gradient update ≠ 永远不更新
```

三者关系：

| 机制 | 当前 backward 有梯度吗 | 参数会变化吗 | 如何变化 |
|---|---:|---:|---|
| 普通可训练网络 | 有 | 会 | optimizer |
| 永久冻结网络 | 无 | 不会 | 不更新 |
| EMA target encoder | 无 | 会 | context 参数的移动平均 |

---

## 4. EMA 公式逐项理解

```math
\xi_t=m\xi_{t-1}+(1-m)\theta_t
```

可以改写为：

```math
\xi_t=\xi_{t-1}+(1-m)(\theta_t-\xi_{t-1})
```

第二种写法更容易看出：

```text
θ_t - ξ_{t-1}：target 与当前 context 的参数差
1 - m：target 每一步追上这段差距的比例
```

例如 `m=0.9`：

```text
target 保留 90% 的旧值
再向当前 context 前进剩余差距的 10%
```

例如 `m=0.996`：

```text
target 保留 99.6% 的旧值
每步只吸收 0.4% 的当前 context 参数
```

---

## 5. 一个标量玩具例子

为便于手算，暂时把整个网络参数压缩成一个标量。

设：

```text
初始 context θ_0 = 0
初始 target  ξ_0 = 0
momentum m = 0.9
```

Optimizer 更新后，context 参数依次变为：

```text
θ_1 = 10
θ_2 = 8
θ_3 = 12
```

第一步：

```math
\xi_1=0.9\times0+0.1\times10=1
```

第二步：

```math
\xi_2=0.9\times1+0.1\times8=1.7
```

第三步：

```math
\xi_3=0.9\times1.7+0.1\times12=2.73
```

观察：

```text
context：0 -> 10 -> 8 -> 12   快速变化
target： 0 ->  1 -> 1.7 -> 2.73 缓慢跟随
```

Target 不等于当前 context，也不是旧 context 的简单复制，而是累积了历史轨迹。

---

## 6. 为什么叫“指数”移动平均

在 momentum 固定时，把递推式展开：

```math
\xi_t=m^t\xi_0+(1-m)\sum_{k=1}^{t}m^{t-k}\theta_k
```

较近的 context 参数权重大，较远的历史权重按 `m` 的幂指数衰减：

```text
当前 θ_t 的权重：       (1-m)
上一步 θ_{t-1} 的权重： (1-m)m
上两步 θ_{t-2} 的权重： (1-m)m²
...
```

这就是 exponential moving average 中“exponential”的含义。

注意：若 `ξ_0=θ_0`，初始项并不是无意义噪声，而是 context encoder 初始参数的一部分历史。

---

## 7. Momentum 决定 target 的记忆长度

### 7.1 `m` 越大，target 越慢

| Momentum `m` | 每步吸收当前 context 的比例 `1-m` | 直觉 |
|---:|---:|---|
| 0 | 100% | target 立即复制 context |
| 0.9 | 10% | 有一定平滑 |
| 0.99 | 1% | 明显慢速 |
| 0.996 | 0.4% | I-JEPA 官方日程起点 |
| 0.999 | 0.1% | 非常缓慢 |
| 1 | 0% | target 完全不再变化 |

### 7.2 有效窗口的粗略直觉

常用近似：

```math
\text{effective window}\approx\frac{1}{1-m}
```

所以：

```text
m=0.9   -> 约 10 steps
m=0.99  -> 约 100 steps
m=0.996 -> 约 250 steps
m=0.999 -> 约 1000 steps
```

这是帮助建立直觉的近似，不代表 EMA 只保留窗口内的参数；更早历史仍有非零但快速衰减的权重。

### 7.3 半衰期

历史权重衰减到一半所需步数约为：

```math
h=\frac{\ln 0.5}{\ln m}
```

`m=0.996` 时，半衰期约为 173 steps。也就是说，一次 context 参数状态在约 173 步后，对 EMA 的相对权重衰减为原来的一半。

---

## 8. I-JEPA 官方动量日程

官方 ViT-H/14 配置写为：

```yaml
optimization:
  ema: [0.996, 1.0]
```

训练代码按总 iteration 数将 momentum 从 `0.996` 线性增加到 `1.0`：

```math
m_t=m_{start}+\frac{t}{T}(m_{end}-m_{start})
```

直觉：

```text
训练早期：target 仍能较快吸收 context 学到的新结构
训练后期：target 更新越来越慢，目标更加稳定
```

最后接近 `1.0` 不等于从一开始就冻结 target。日程中绝大部分训练步骤的 `m_t` 仍小于 1。

对于小型实验，不要盲目照抄大规模训练的最优值。训练步数、batch size、学习率与模型规模变化后，EMA 的有效时间尺度也会变化。

---

## 9. 正确的一次训练 step

概念顺序：

```text
1. 用当前 ξ 前向计算 z_target，且不建立梯度
2. 用当前 θ 和 φ 前向计算 z_pred
3. 计算 SmoothL1/L1 latent loss
4. backward
5. optimizer 更新 θ 和 φ
6. 用更新后的 θ 对 ξ 做 EMA
7. 进入下一批数据
```

I-JEPA 官方实现的核心结构可以压缩为：

```python
with torch.no_grad():
    z_target = target_encoder(images)
    z_target = gather_target_tokens(z_target, target_masks)

z_context = context_encoder(images, context_masks)
z_pred = predictor(z_context, context_masks, target_masks)
loss = smooth_l1_loss(z_pred, z_target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

with torch.no_grad():
    for p_context, p_target in zip(
        context_encoder.parameters(),
        target_encoder.parameters(),
    ):
        p_target.mul_(momentum).add_(
            p_context,
            alpha=1.0 - momentum,
        )
```

这里的 `z_target` 是 EMA 更新前的 target encoder 输出，而下一 step 才会使用刚更新后的 target 参数。

---

## 10. 初始化与参数配对

### 10.1 Target encoder 如何初始化

官方代码先创建 context encoder，再深拷贝：

```python
target_encoder = copy.deepcopy(context_encoder)
```

因此初始时：

```text
ξ_0 = θ_0
```

若两个 encoder 独立随机初始化，训练一开始 predictor 需要对齐一个完全不同的随机映射，增加无意义的困难。

### 10.2 参数必须一一对应

EMA 假设两个网络：

- 架构相同；
- 参数数量相同；
- 参数顺序与 shape 一一对应。

至少应检查：

```python
for p_context, p_target in zip(
    context_encoder.parameters(),
    target_encoder.parameters(),
):
    assert p_context.shape == p_target.shape
```

如果修改了其中一个 encoder 的层结构，简单 `zip` 可能静默错配或漏掉参数。

---

## 11. EMA 为什么能让目标更稳定

假设 context encoder 因不同 mini-batch 的梯度来回波动：

```text
θ：向左 -> 向右 -> 向左 -> 向右
```

EMA 会削弱短期高频变化：

```text
ξ：缓慢向整体趋势移动
```

因此 predictor 面对的 target representations 不会在相邻 steps 中剧烈跳动。

稳定目标的意义是：

1. 当前 student 有一个相对固定的参照；
2. target 聚合了多个历史 student 状态；
3. 优化不必同时追逐两条快速移动的分支；
4. 自蒸馏过程具有时间一致性。

但 target 不是外部真值。它最终仍来源于 context encoder，因此属于 bootstrap/self-distillation。

---

## 12. EMA 与表征坍塌的准确关系

错误说法：

```text
“只要用了 EMA，模型就不可能 collapse。”
```

考虑常数表示：

```math
f_\theta(x)=c,\qquad f_\xi(x)=c,\qquad g_\phi(c,pos)=c
```

此时 latent matching loss 仍可能为零，EMA 也会让两个 encoder 继续保持相似的常数映射。

更准确的说法：

> EMA 与 stop-gradient 建立慢 target、快 student 的不对称训练动力学，提供稳定参照；I-JEPA 的非坍塌学习还依赖 predictor、mask 任务、架构、归一化、优化过程和数据变化。

因此仍需监测：

- batch feature standard deviation；
- pairwise cosine similarity；
- covariance spectrum / effective rank；
- linear probe 或 kNN 表现。

低 loss 不能单独证明 representation 有用。

---

## 13. 常见实现错误

### 错误 1：把 target encoder 放进 optimizer

```python
optimizer = AdamW(
    list(context.parameters())
    + list(predictor.parameters())
    + list(target.parameters())  # 错误
)
```

后果：target 同时受到 gradient 与 EMA 两种更新，角色定义被破坏。

### 错误 2：忘记 `no_grad` 或 `requires_grad=False`

即使最后没有调用 target 的 optimizer step，也会无谓地构建计算图、占用显存，并增加误更新风险。

### 错误 3：EMA 方向写反

错误：

```python
context.mul_(m).add_(target, alpha=1-m)
```

正确：

```python
target.mul_(m).add_(context, alpha=1-m)
```

被平滑更新的是 target，不是 context。

### 错误 4：使用错误的系数

错误：

```python
target = (1-m) * target + m * context
```

当 `m=0.996` 时，这会让 target 每步吸收 99.6% 的 context，几乎失去慢速作用。

### 错误 5：每个 epoch 才更新一次 EMA

官方实现按 training iteration 更新。若改成每个 epoch 更新一次，相同 `m` 对应的真实时间尺度完全不同。

### 错误 6：断点恢复后重置 target 或 momentum 日程

恢复训练时应同时恢复：

```text
context encoder
predictor
target encoder
optimizer/scaler
当前 epoch/iteration 对应的 scheduler 状态
```

若只加载 context encoder，再重新复制 target，EMA 历史会被清空。

### 错误 7：只验证参数接近，不验证梯度

Target 与 context 参数接近是预期现象，但还必须检查 target gradients 为 `None`。

---

## 14. 三个立即练习

### 练习 1：手算 EMA

已知：

```text
ξ_old = 2
θ_new = 10
m = 0.75
```

求 `ξ_new`。

提示：旧 target 权重为 `m`，当前 context 权重为 `1-m`。

<details>
<summary>答案</summary>

```math
\xi_{new}=0.75\times2+0.25\times10=4
```

</details>

### 练习 2：判断更新顺序

下面哪一个更接近 I-JEPA 官方训练流程？

```text
A. EMA target -> forward -> loss -> optimizer
B. forward -> loss -> optimizer context/predictor -> EMA target
C. forward -> loss -> optimizer all three networks
```

<details>
<summary>答案</summary>

`B`。当前 step 用旧 target 产生监督；optimizer 更新 context/predictor 后，再让 target 缓慢跟随更新后的 context。

</details>

### 练习 3：找三个 bug

```python
target = TargetEncoder()
optimizer = AdamW(
    list(context.parameters()) + list(target.parameters())
)

z_target = target(x)
loss = loss_fn(predictor(context(x)), z_target)
loss.backward()
optimizer.step()

with torch.no_grad():
    for q, k in zip(context.parameters(), target.parameters()):
        q.mul_(m).add_(k, alpha=1-m)
```

<details>
<summary>答案</summary>

至少有四个问题：

1. target 没有从 context 深拷贝初始化；
2. target 被放进 optimizer；
3. target forward 没有 stop-gradient / `no_grad`；
4. EMA 方向写反，代码更新了 context 参数。

</details>

---

## 15. 为 mini-I-JEPA 写一个 EMA 单元测试

建议在真正训练前验证四个不变量：

```python
import copy
import torch

context = TinyEncoder()
target = copy.deepcopy(context)

for p in target.parameters():
    p.requires_grad = False

# 1. 初始化完全相同
for p_context, p_target in zip(
    context.parameters(), target.parameters()
):
    assert torch.equal(p_context, p_target)

# 2. 只让 context 发生一个人工变化
with torch.no_grad():
    for p in context.parameters():
        p.add_(1.0)

# 3. EMA 后 target 应位于旧 target 与新 context 之间
m = 0.9
with torch.no_grad():
    for p_context, p_target in zip(
        context.parameters(), target.parameters()
    ):
        old_target = p_target.clone()
        expected = m * old_target + (1.0 - m) * p_context
        p_target.mul_(m).add_(p_context, alpha=1.0 - m)
        assert torch.allclose(p_target, expected)

# 4. Target 不应有 optimizer gradient
for p in target.parameters():
    assert p.grad is None
```

训练日志中可额外记录 encoder 参数距离：

```math
d_t=\frac{1}{P}\sum_{i=1}^{P}
\|\theta_t^{(i)}-\xi_t^{(i)}\|_2
```

它不应被机械追求为零。距离太大可能说明 target 跟不上；始终严格为零则可能表示 EMA 没有真正形成慢分支，或 context 根本没有更新。

---

## 16. 迁移规则：什么时候想到 EMA teacher

抽象模式：

```text
学习目标由另一个正在学习的网络产生
        ↓
当前目标若变化太快，student 难以稳定追踪
        ↓
stop-gradient + EMA 构造慢速目标网络
```

你会在这些方法中看到类似思想：

- BYOL：online network 预测 EMA target network；
- DINO：student 匹配 EMA teacher 的输出分布；
- I-JEPA：context/predictor 预测 EMA target encoder 的 patch representations；
- Mean Teacher：学生预测与 teacher 预测保持一致。

识别规则：

> 当监督信号来自模型自身，而不是固定标签时，先检查谁接收梯度、谁提供目标、目标以什么速度更新。

边界：EMA 适合稳定自生成目标，但不能自动修复错误目标、数据偏差、信息泄漏或不合理的预测任务。

---

## 17. 今日理解检查

不看上文回答：

1. 写出 `ξ_t = mξ_{t-1} + (1-m)θ_t`，并解释每一项。
2. 为什么 `m` 越接近 1，target encoder 更新越慢？
3. `m=0` 与 `m=1` 分别发生什么？
4. Stop-gradient 与永久冻结有什么区别？
5. Target encoder 为什么通常从 context encoder 深拷贝初始化？
6. Optimizer 应更新哪两个模块？
7. EMA 应发生在 optimizer step 之前还是之后？
8. 为什么 EMA 不能单独保证不坍塌？
9. 断点恢复时为什么必须加载 target encoder 状态？
10. 只看训练 loss 为什么无法判断 target 表征是否有用？

完成标准：

- [ ] 能闭卷写出 EMA 公式；
- [ ] 能手算三步标量 EMA；
- [ ] 能解释 momentum 与有效记忆长度的关系；
- [ ] 能画出 gradient update 与 EMA update 两条路径；
- [ ] 能写出一次正确训练 step 的顺序；
- [ ] 能发现 optimizer 包含 target、EMA 方向写反等 bug；
- [ ] 不再把 EMA teacher 说成固定老师或防坍塌定理。

---

## 18. 不看笔记时应该能复述的版本

### 30 秒版本

I-JEPA 的 context encoder 和 predictor 由 optimizer 更新，target encoder 不接收梯度，而是在每个训练 step 后按 `ξ_t=m_tξ_{t-1}+(1-m_t)θ_t` 缓慢跟随 context encoder。这样 target 表征聚合了 student 的历史状态，不会随单个 batch 剧烈变化，为 latent prediction 提供更稳定的目标。官方 ViT-H/14 配置把 momentum 从 0.996 线性增加到 1.0。EMA 与 stop-gradient 建立训练不对称，但不能单独保证 representation 不坍塌。

### 五行记忆版

```text
Context encoder：梯度更新，变化快。
Target encoder：没有梯度，EMA 更新，变化慢。
ξ <- mξ + (1-m)θ。
m 越大，历史越长，target 越稳定。
EMA 稳定目标，但不是防坍塌魔法。
```

### 最短主线

```text
快速 learner + 缓慢 target = 可追踪的自生成监督信号
```

---

## 19. 与 Day 10 的衔接：Predictor 为什么需要 target position

现在已经知道：

```text
Mask sampler 决定猜哪里；
EMA target encoder 产生稳定的 latent 答案。
```

下一步要回答：

```text
相同 z_context 对应多个 target blocks 时，predictor 如何区分它们？
target position embedding 如何进入 predictor？
为什么“知道位置”不等于“看到 target pixels”？
z_context + target_pos 如何变成 z_pred？
```

---

## 20. 今日学习日志

```text
计划日期：2026-07-17
实际接续日期：2026-07-20
主题：EMA target encoder

今天学到的核心概念：

我能用自己的话解释 EMA：

我能手写的更新公式：

m 变大时会发生什么：

Optimizer 更新的模块：

EMA 更新的模块：

我还不清楚的点：
1.
2.

今天的产出文件：
notes/day09_ema_target_encoder.md

下一步：
学习 predictor，画出 z_context + target_position -> z_pred。
```
