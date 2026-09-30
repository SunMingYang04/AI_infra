# AI Systems Engineer 项目驱动学习路线
## LLM Infra / GPU Optimization / Kernel Agent — Project-Driven Edition

> 适用目标：AI Systems Engineer / LLM Inference / GPU Kernel / AI Infra  
> 核心思想：**不是先学完知识再做项目，而是用项目暴露知识缺口，再针对缺口补课。**  
> 原技术依赖链保留：**系统基础 → CUDA → Triton → LLM 推理 → 性能工程 → Agent 自动化**  
> 学习模式：**Project → Issue → Knowledge Gap → Learn → Implement → Benchmark/Profile → Explain → Commit**

---

# 0. 这份路线怎么用

这不是一份“每天看第几章”的课程表。

你之后的主要学习单位不再是：

- Chapter
- Lecture
- Day
- Week

而是：

- Project
- Milestone
- Issue
- Bug
- Performance bottleneck
- Pull Request / Commit
- Benchmark result

以后每次学习都从一个明确的问题开始：

```text
我现在要实现什么？
        ↓
我不会什么？
        ↓
最小需要补哪些知识？
        ↓
查资料 / 看指定章节
        ↓
回来实现
        ↓
跑正确性测试
        ↓
Benchmark / Profile
        ↓
解释为什么
        ↓
Commit
```

## 0.1 三条最高优先级规则

### Rule 1：项目优先，课程按需

教材、课程、论文不再是主线。

它们是工具箱。

只有当项目暴露出明确知识缺口时，才去读对应部分。

例如：

```text
Buffer 出现 double free
→ 补 copy constructor / move semantics

CUDA transpose 很慢
→ 补 coalescing / shared memory / bank conflict

Decode 越来越慢
→ 补 KV Cache

FlashAttention 看不懂
→ 补 online softmax + memory hierarchy
```

---

### Rule 2：允许提前查知识，不允许无目的跳阶段

你可以因为当前项目需要提前查某个知识点。

但不允许因为“这个东西看起来很有趣”就跳到后面的阶段。

例如：

允许：

```text
CUDA GEMM
→ 为理解性能补 Roofline
```

不允许：

```text
CUDA GEMM 还没写明白
→ 突然去研究 vLLM scheduler
→ 又去学 LangGraph
```

---

### Rule 3：所有项目必须形成可验证产出

每个项目至少要有：

```text
README
代码
运行命令
正确性测试
Benchmark
Profile 结果
问题记录
关键结论
Git commit 历史
```

目标不是“我学过”。

目标是：

> **我做过、测过、解释得出来。**

---

# 1. 总体项目链

整条路线只围绕下面 6 个核心项目推进。

```text
P0 cpp-systems-lab
        ↓
P1 cuda-kernel-lab
        ↓
P2 triton-kernel-lab
        ↓
P3 mini-llm-runtime + vllm-hacking
        ↓
P4 llama-perf-report
        ↓
P5 AutoKernel-Agent
```

它们最终形成一条完整作品集叙事：

```text
系统基础
↓
手写 CUDA Kernel
↓
用 Triton 重构 Kernel
↓
把 Kernel 放进 LLM Runtime
↓
对完整推理系统做性能优化
↓
让 Agent 自动执行优化闭环
```

---

# 2. 统一学习闭环

之后所有阶段都遵守同一个循环。

## Step 1：定义当前 Issue

每次只解决一个明确问题。

例：

```text
Issue #03
实现一个固定大小 ThreadPool，
支持提交 1000 个独立任务，
并比较 1/2/4/8 threads 的执行时间。
```

不要同时开五个方向。

---

## Step 2：先尝试实现

在查完整答案之前，先写。

目标不是第一次写对，而是：

> 让自己的真实知识缺口暴露出来。

---

## Step 3：记录知识缺口

统一使用：

```markdown
## Knowledge Gaps

- [ ] 不理解 condition_variable 为什么需要 predicate
- [ ] 不理解 spurious wakeup
- [ ] 不清楚 mutex 应保护什么数据
```

---

## Step 4：最小补课

只学习解决当前问题所需的最小知识。

推荐顺序：

```text
官方文档
→ 教材指定章节
→ 高质量课程/博客
→ 源码
→ 论文
```

不要为了一个问题完整看完一本书。

---

## Step 5：完成实现

必须能运行。

---

## Step 6：验证正确性

根据项目类型使用：

```text
assert
unit test
allclose
reference implementation
sanitizer
race detector
```

---

## Step 7：Benchmark / Profile

只要项目涉及性能，就必须测。

```text
Baseline
↓
修改
↓
重新测
↓
对比
```

---

## Step 8：解释结果

必须回答：

```text
为什么变快？
为什么变慢？
瓶颈在哪里？
我的证据是什么？
```

---

## Step 9：Git Commit

建议 commit 信息表达“工程变化”，而不是“学习进度”。

推荐：

```text
feat: add RAII buffer implementation
fix: implement move constructor to avoid double free
perf: use shared-memory tiling for matrix transpose
bench: add KV-cache decode benchmark
```

不推荐：

```text
day2
study cpp
finished lesson
```

---

# 3. P0 — cpp-systems-lab

## 3.1 项目目标

不是“学完 C++ 和操作系统”。

而是通过几个小系统组件建立：

- C++ 对象生命周期直觉
- 内存模型
- 线程同步
- Linux process / syscall
- mmap / IPC
- Python ↔ C++ 扩展接口

最终仓库：

```text
cpp-systems-lab/
├── memory_layout/
├── buffer/
├── thread_pool/
├── linux_lab/
├── numa_lab/
├── pybind_ext/
├── notes/
└── README.md
```

---

## 3.2 Milestone 0.1 — Memory Layout

### Issue

写一个程序观察：

```text
global variable
stack variable
heap variable
class object
static variable
```

打印地址、`sizeof`、alignment。

### 你可能遇到的知识缺口

- stack vs heap
- object layout
- alignment / padding
- pointer / reference
- virtual function table

### 按需资料

优先查：

- C++ Primer：类型、数组、类
- CSAPP：Memory Hierarchy
- gcc `-fstack-usage`
- gdb

### 交付物

```text
memory_layout/
├── main.cpp
├── Makefile/CMakeLists.txt
└── README.md
```

### 验收

你能解释：

1. stack 和 heap 的差异；
2. `sizeof(struct)` 为什么可能大于成员大小之和；
3. pointer 存的是什么；
4. object 与 object 内部 buffer 的地址为什么不同。

---

# 4. P0.2 — 自己实现 Buffer

这是整个路线真正的起点。

## v1 — Raw Buffer

目标：

```cpp
Buffer(size_t n)
```

支持：

- heap allocation
- size
- read/write
- destructor

---

## v2 — Copy

执行：

```cpp
Buffer a(1024);
Buffer b = a;
```

观察行为。

你很可能会遇到：

```text
double free
shallow copy
ownership
```

### 触发学习

- copy constructor
- copy assignment
- deep copy
- Rule of 3

---

## v3 — Move

支持：

```cpp
Buffer create_buffer();
Buffer b = create_buffer();
```

学习：

- lvalue / rvalue
- move constructor
- move assignment
- Rule of 5

---

## v4 — RAII

重新设计资源所有权。

理解：

```text
resource acquisition
=
object lifetime
```

---

## v5 — Smart Pointer

比较：

```text
raw pointer
unique_ptr
shared_ptr
```

---

## 最终验收

你必须能脱稿解释：

```text
为什么 C++ 需要 RAII？
什么情况下发生 double free？
copy 与 move 有什么本质差异？
unique_ptr 为什么不能 copy？
```

---

# 5. P0.3 — Thread Pool

## 项目目标

实现：

```text
Fixed-size ThreadPool
```

支持：

- N worker threads
- task queue
- submit task
- graceful shutdown

---

## 项目推进顺序

### v1

直接：

```cpp
std::thread
```

启动多个线程。

### v2

多个线程读取同一个 queue。

大概率出现：

```text
race condition
```

### v3

加入：

```text
mutex
```

### v4

避免 worker 空转：

```text
condition_variable
```

### v5

支持返回值：

```text
future / promise
```

---

## Knowledge Trigger

遇到什么学什么：

| 项目问题 | 补知识 |
|---|---|
| queue 数据错乱 | race condition / mutex |
| CPU 100% | busy waiting |
| worker 如何休眠 | condition_variable |
| submit 如何返回结果 | future / promise |
| 偶发死锁 | lock ordering |

---

## Benchmark

运行：

```text
1000 tasks

1 thread
2 threads
4 threads
8 threads
```

记录：

```text
runtime
speedup
CPU utilization
```

---

# 6. P0.4 — Linux Lab

## Issue A — fork

写：

```text
parent
↓ fork
child
```

观察：

- PID
- address
- variable value

触发学习：

```text
process
virtual memory
copy-on-write
```

---

## Issue B — Pipe

实现：

```text
cmd1 | cmd2
```

理解：

```text
file descriptor
pipe
dup2
fork
exec
```

---

## Issue C — mmap

实现一个大文件 copy：

```text
read/write version
vs
mmap version
```

---

## 工具

必须实际使用：

```text
strace
lsof
perf
```

---

# 7. P0.5 — Python ↔ C++

## 项目目标

写一个简单算子：

```python
y = custom_add(x)
```

Python 调用 C++。

路线：

```text
C++
↓
pybind11
↓
torch.utils.cpp_extension
↓
Python
```

这个项目会成为之后 CUDA Extension 的接口基础。

---

# 8. P0 出关标准

不要看“学了几周”。

只看你能否完成这些事情：

- [ ] 能自己解释对象生命周期；
- [ ] 能解决 double free；
- [ ] 能写 ThreadPool；
- [ ] 能解释 mutex / condition_variable；
- [ ] 能使用 fork / pipe / mmap；
- [ ] 能使用 strace；
- [ ] 能让 Python 调用 C++。

如果这些做不到，不进入 CUDA。

---

# 9. P1 — cuda-kernel-lab

## 9.1 核心目标

从：

> “CUDA 是什么？”

推进到：

> “我能根据 profiler 判断 kernel 为什么慢，并进行优化。”

仓库：

```text
cuda-kernel-lab/
├── vector_add/
├── transpose/
├── reduction/
├── layernorm/
├── gemm/
├── benchmarks/
├── profiles/
└── README.md
```

---

# 10. P1.1 — Vector Add

第一版直接写：

```text
C = A + B
```

不要提前系统看完整 CUDA Programming Guide。

先跑起来。

---

## 第一个核心问题

CPU 版：

```text
time = ?
```

GPU 版：

```text
time = ?
```

你可能发现：

> GPU 不一定明显更快。

这时候再学习：

```text
kernel launch overhead
host ↔ device transfer
thread
block
grid
```

---

## 必做实验

改变：

```text
block size = 32
64
128
256
512
```

记录 latency。

---

# 11. P1.2 — Matrix Transpose

实现：

```text
naive transpose
```

然后 profile。

你会遇到：

```text
uncoalesced memory access
```

再学习：

```text
global memory
coalescing
shared memory
```

---

## v2

加入 shared memory tile。

再次 benchmark。

---

## v3

观察：

```text
bank conflict
```

然后补：

```text
shared memory bank
padding
```

---

# 12. P1.3 — Reduction

实现：

```text
sum(x)
```

从最朴素版本开始。

逐步遇到：

```text
branch divergence
synchronization
warp
```

然后学习：

```text
warp
SIMT
warp divergence
__syncthreads
warp-level primitive
```

---

# 13. P1.4 — LayerNorm

目标：

```text
mean
variance
normalize
affine
```

这里开始组合：

```text
reduction
memory access
fusion
```

并第一次认真使用：

```text
Nsight Compute
```

---

# 14. P1.5 — GEMM Optimization Ladder

这是 P1 的核心项目。

## v1 — Naive

```text
one thread
→ one C element
```

记录：

```text
TFLOPS
latency
```

---

## v2 — Coalescing

调整访存方式。

---

## v3 — Shared Memory Tiling

学习：

```text
tiling
data reuse
```

---

## v4 — Register Tiling

减少：

```text
shared memory traffic
```

---

## v5 — Vectorized Load

---

## v6 — Double Buffering

---

## v7 — Tensor Core / WMMA

最后才进入 Tensor Core。

---

## 每个版本都记录

```markdown
| Version | Latency | TFLOPS | Bottleneck | Key Change |
|---|---:|---:|---|---|
| v1 | | | | |
| v2 | | | | |
```

并与：

```text
cuBLAS
```

对比。

---

# 15. P1 必学工具

项目触发后学习：

```text
nsys
ncu
CUDA Event
Roofline
```

不要先学工具命令大全。

只学习：

> 当前 profile 需要哪些指标。

---

# 16. P1 出关标准

你必须可以拿一个陌生 kernel 回答：

```text
它为什么慢？
```

并从这些维度检查：

- memory bandwidth
- compute
- occupancy
- memory coalescing
- bank conflict
- divergence
- launch overhead

必须完成：

- [ ] Vector Add
- [ ] Transpose
- [ ] Reduction
- [ ] LayerNorm
- [ ] GEMM optimization ladder
- [ ] 至少一次完整 Nsight 分析

---

# 17. P2 — triton-kernel-lab

## 核心思想

不要从头学习 Triton。

直接把你已经写过的 CUDA kernel 用 Triton 重写。

这样你天然有比较对象。

---

# 18. P2.1 — Vector Add

CUDA：

```text
threadIdx
blockIdx
```

Triton：

```text
program_id
arange
load
store
mask
```

比较两种 programming model。

---

# 19. P2.2 — Softmax

先写 PyTorch baseline：

```python
torch.softmax
```

然后 Triton。

遇到 reduction 再学：

```text
tl.max
tl.sum
```

---

# 20. P2.3 — Matmul

把 CUDA GEMM 的理解迁移过来：

```text
BLOCK_M
BLOCK_N
BLOCK_K
```

然后：

```text
autotune
```

---

# 21. P2.4 — LayerNorm / RMSNorm

最终 RMSNorm 要为后续 LLM Runtime 服务。

---

# 22. P2.5 — FlashAttention

这是整个 P2 的核心。

不要直接背论文。

先实现标准 attention：

```text
QK^T
↓
softmax
↓
PV
```

记录显存。

---

## 然后问

为什么：

```text
Attention Matrix
=
O(N²)
```

---

## 再学习

```text
online softmax
tiling
SRAM reuse
IO-aware algorithm
```

---

## 最后实现

```text
Triton FlashAttention
```

并比较：

```text
PyTorch SDPA
```

测试：

```text
seq_len = 512
2048
8192
```

记录：

```text
latency
peak memory
```

---

# 23. P2 出关标准

你必须能解释：

1. Triton 与 CUDA programming model 的区别；
2. 为什么 Triton 可以隐藏 thread-level management；
3. FlashAttention 为什么快；
4. 为什么显存复杂度降低；
5. online softmax rescale 是什么。

---

# 24. P3 — mini-llm-runtime

这是整条路线的中心项目。

目标不是“学 Transformer”。

目标是：

> **自己写一个能生成 token 的最小 LLM Runtime。**

---

# 25. P3 仓库结构

```text
mini-llm-runtime/
├── model/
├── attention/
├── kernels/
├── runtime/
├── tokenizer/
├── benchmarks/
└── README.md
```

---

# 26. Milestone 3.1 — 加载模型

选一个小模型，例如：

```text
Qwen 0.5B
或其他小型 decoder-only model
```

完成：

```text
checkpoint load
tokenizer
embedding
```

---

# 27. Milestone 3.2 — Transformer Block

逐步实现：

```text
RMSNorm
↓
QKV projection
↓
Attention
↓
Output projection
↓
Residual
↓
MLP
↓
Residual
```

项目卡住时再补：

```text
Transformer
MHA
GQA
RoPE
RMSNorm
```

---

# 28. Milestone 3.3 — Naive Decode

先不使用 KV Cache。

```text
token 1
↓
recompute all history

token 2
↓
recompute all history
```

Benchmark：

```text
sequence length
vs
latency/token
```

---

# 29. Milestone 3.4 — KV Cache

然后加入：

```text
K cache
V cache
```

比较：

```text
without KV cache
vs
with KV cache
```

你会自然理解：

```text
为什么 KV Cache 存在
```

---

# 30. Milestone 3.5 — Prefill vs Decode

分别 benchmark：

```text
prefill
decode
```

然后用：

```text
arithmetic intensity
Roofline
```

解释：

```text
prefill → more compute-bound
decode → more memory-bound
```

---

# 31. Milestone 3.6 — GQA

从：

```text
MHA
```

升级：

```text
GQA
```

观察：

```text
KV Cache memory
```

变化。

---

# 32. Milestone 3.7 — 接入自己的 Triton Kernel

至少把一个算子换成：

```text
P2 自己写的 Triton Kernel
```

例如：

```text
RMSNorm
```

验证：

```python
torch.allclose(...)
```

---

# 33. P3-B — vllm-hacking

有了自己的 runtime 后再读 vLLM。

否则很容易变成：

> 看懂类名，但不知道为什么系统需要这些模块。

---

## 阅读路径

带着问题读：

```text
一个 request
到底如何从输入
变成 GPU 上执行的一轮 decode？
```

关注：

```text
engine
scheduler
KV cache manager
attention backend
worker
model executor
```

---

## 至少修改一个东西

任选：

```text
scheduler logging
KV cache metrics
attention backend
custom kernel
```

留下自己的 commit。

---

# 34. P3 出关标准

必须能从代码层面解释：

```text
prompt
↓
tokenize
↓
prefill
↓
KV Cache
↓
decode
↓
sampling
↓
next token
```

同时能解释：

- KV Cache
- PagedAttention
- Continuous Batching
- Prefill
- Decode
- MHA / MQA / GQA

---

# 35. P4 — llama-perf-report

这一阶段不再以“实现功能”为核心。

而是：

> **性能工程。**

你要优化 P3 的 Runtime。

---

# 36. 性能工程闭环

必须严格遵守：

```text
Benchmark
↓
Profile
↓
Identify Bottleneck
↓
Form Hypothesis
↓
Change ONE Variable
↓
Benchmark Again
↓
Correctness Regression
↓
Explain
```

---

# 37. Baseline

记录：

```text
TTFT
TPOT
tokens/s
GPU memory
GPU utilization
```

---

# 38. Optimization Ladder

逐级加入：

## O1 — KV Cache

记录前后差异。

## O2 — FlashAttention

记录：

```text
latency
memory
```

## O3 — Quantization

例如：

```text
INT8 / FP8 / AWQ / GPTQ
```

## O4 — vLLM

比较：

```text
your runtime
vs
vLLM
```

---

# 39. 每次优化必须产生

```text
Before
After
Profile evidence
Mechanism explanation
Correctness evidence
```

---

# 40. 最终 Performance Report

README / report 至少包括：

```text
Environment
Benchmark methodology
Baseline
Profiler screenshots
Optimization hypotheses
Before/After
Roofline
Correctness
Remaining bottlenecks
```

---

# 41. P4 出关标准

别人给你一个慢的 LLM inference pipeline。

你必须能说：

```text
先测什么
再看什么
怎么判断
怎么优化
怎么验证
```

而不是：

> 换 vLLM 应该就会快。

---

# 42. P5 — AutoKernel-Agent

只有前面全部做过，才进入 Agent。

原因：

> Agent 应该自动化你已经理解的流程，而不是替代你没有掌握的知识。

---

# 43. AutoKernel-Agent 的目标

输入：

```text
baseline kernel
```

系统自动执行：

```text
Analyze
↓
Compile
↓
Correctness Check
↓
Benchmark
↓
Profile
↓
Diagnose
↓
Generate Candidate
↓
Compile
↓
Benchmark
↓
Compare
↓
Iterate
```

---

# 44. 开发顺序

不要先搭 Agent 框架。

顺序必须是：

```text
runner
↓
benchmark
↓
correctness
↓
profiler
↓
diagnosis
↓
LLM generator
↓
agent loop
```

---

# 45. Milestone 5.1 — Runner

实现：

```text
compile
run
timeout
capture stdout/stderr
```

---

# 46. Milestone 5.2 — Correctness Gate

候选 kernel 必须：

```text
allclose(reference)
```

否则直接淘汰。

---

# 47. Milestone 5.3 — Benchmark

统一计时口径。

避免：

```text
fake speedup
```

---

# 48. Milestone 5.4 — Profiler

自动调用：

```text
ncu
nsys
Proton
```

提取：

```text
bandwidth
occupancy
warp stall
bank conflict
```

---

# 49. Milestone 5.5 — Diagnosis

先规则化：

```text
if bandwidth high
and FLOPS low
→ memory bound
```

不要一开始所有判断都交给 LLM。

---

# 50. Milestone 5.6 — Candidate Generator

LLM 输入：

```text
source code
benchmark
profile summary
previous attempts
```

输出：

```text
candidate kernel
optimization rationale
```

---

# 51. Milestone 5.7 — Multi-round Loop

最后才引入：

```text
LangGraph / Agent Framework
```

状态：

```text
baseline
candidate
metrics
history
best_latency
iteration
```

---

# 52. 最终验收

至少在 3 个任务上：

```text
baseline
↓
agent optimization
↓
correct output
↓
measured speedup
```

并保留：

```text
每轮代码
每轮 latency
每轮 diagnosis
```

---

# 53. 资料系统：从“课程列表”改成“知识索引”

教材不再规定完成时间。

改为按问题调用。

---

## C++

遇到这些问题时查：

```text
pointer/reference
RAII
copy/move
template
thread
mutex
condition_variable
```

资料：

- C++ Primer
- Effective Modern C++
- C++ Concurrency in Action

---

## Systems

问题：

```text
process
syscall
virtual memory
mmap
page fault
IPC
NUMA
```

资料：

- CSAPP
- OSTEP
- Linux man-pages

---

## CUDA

问题：

```text
thread/block
memory hierarchy
coalescing
shared memory
occupancy
warp divergence
Tensor Core
```

资料：

- CUDA Programming Guide
- CUDA Best Practices
- PMPP
- GPU MODE

---

## Triton

问题：

```text
program_id
tile
mask
tl.dot
autotune
FlashAttention
```

资料：

- Triton Tutorials
- Triton API
- Triton Puzzles
- FlashAttention papers

---

## LLM Inference

问题：

```text
KV Cache
PagedAttention
Continuous Batching
GQA
quantization
tensor parallel
```

资料：

- Attention Is All You Need
- nanoGPT
- vLLM paper
- Orca
- vLLM source

---

# 54. 每周管理方式

不再规定：

```text
Week 1 看什么
Week 2 看什么
```

每周只定义：

```text
1 个主 Milestone
1–3 个 Issues
1 个可运行交付物
```

---

## 推荐 Weekly Template

```markdown
# Week X

## Main Milestone

实现：

...

## Issues

- [ ] Issue 1
- [ ] Issue 2
- [ ] Issue 3

## Knowledge Gaps

- [ ] ...
- [ ] ...

## Learn on Demand

- ...
- ...

## Benchmark

Baseline:

After:

## What I Learned

1.
2.
3.

## Remaining Problems

1.
2.

## Git Commits

- ...
```

---

# 55. Daily 工作方式

一天学习不要按：

```text
今天看 3 小时视频
```

记录。

使用：

```text
今天关闭了几个 Issue？
```

例如：

```text
Today

Issue:
ThreadPool worker busy waiting

Goal:
使用 condition_variable 消除 busy waiting

Result:
CPU idle utilization 从 ~100% 降低

Knowledge Gap:
condition_variable predicate

Commit:
perf: replace busy-wait loop with condition variable
```

---

# 56. AI 导师使用方式

以后向 AI 提问时不要：

```text
教我 CUDA
```

而应该：

```text
这是我的 transpose kernel。
当前 global load efficiency 很低。
请先不要直接给优化后代码。

1. 帮我判断最可能的原因；
2. 告诉我需要补什么知识；
3. 给我一个实验验证；
4. 等我修改后再 review。
```

这样 AI 才是：

```text
mentor
```

而不是：

```text
answer generator
```

---

# 57. 卡住时的标准处理协议

如果一个 Issue 卡住：

## < 30 min

自己 debug。

## 30–90 min

查官方文档 / 教材。

## > 90 min

向 AI 提问，但必须提供：

```text
代码
错误
预期行为
实际行为
已经尝试过什么
```

---

# 58. 不允许的学习方式

以后尽量避免：

### 1. 收藏大量课程

```text
收藏 ≠ 学习
```

### 2. 同时学习多个大方向

```text
CUDA
Triton
vLLM
Agent
```

同时推进会导致全部停留在入门。

### 3. 为了“准备好了”一直不写代码

你永远不会完全准备好。

### 4. 只追求代码跑通

AI Infra 更关键的是：

```text
为什么慢？
为什么快？
```

### 5. 只看平均性能

要建立 benchmark discipline。

---

# 59. 项目验收优先于时间

原计划中的周数现在只作为参考。

真正推进条件：

```text
Milestone Passed
→ Next Project
```

而不是：

```text
Week Finished
→ Next Project
```

---

# 60. 最终作品集

最终 GitHub 建议形成：

```text
01-cpp-systems-lab
02-cuda-kernel-lab
03-triton-kernel-lab
04-mini-llm-runtime
05-vllm-hacking
06-llama-perf-report
07-AutoKernel-Agent
```

每个仓库都应该体现：

```text
Problem
↓
Baseline
↓
Implementation
↓
Bottleneck
↓
Optimization
↓
Measurement
↓
Conclusion
```

---

# 61. 求职叙事

最终你应该能够用一条连续技术故事介绍自己：

> 我先从系统编程和内存模型开始，自己实现了线程池和 Python/C++ 扩展；随后进入 CUDA，围绕 transpose、reduction、LayerNorm 和 GEMM 做 kernel 优化，并使用 Nsight 和 Roofline 做性能分析；之后用 Triton 重写关键 kernel，并实现 FlashAttention；在此基础上写了一个最小 LLM Runtime，实现 Prefill、Decode、KV Cache 和 GQA，并阅读和修改 vLLM；随后对完整 LLM inference pipeline 进行 benchmark 和性能优化；最后把这套 profile → diagnose → optimize → benchmark 的流程自动化成 AutoKernel-Agent。

这条叙事比：

> 我学过 CUDA、Triton、vLLM。

有明显更高的信息密度。

---

# 62. 当前起点

现在不再继续传统的 Stage 0 Day 2。

正式进入：

```text
Project 0
cpp-systems-lab

Milestone 0.1
Buffer
```

第一阶段：

```text
Issue #1
实现最小 Buffer
```

要求：

```text
1. heap 上申请 N bytes
2. 保存 size
3. 支持简单读写
4. destructor 释放内存
5. 打印 object address
6. 打印 data address
```

当前不要求：

```text
copy
move
smart pointer
template
```

这些会在项目真正需要时逐步加入。

---

# 63. 总原则

整条路线最终只遵守一句话：

> **Build first. Learn when blocked. Measure everything. Explain why.**

中文：

> **先做，遇到问题再学；所有性能结论必须测；所有优化必须能解释。**
