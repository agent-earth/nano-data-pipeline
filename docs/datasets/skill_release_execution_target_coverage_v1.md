# Skill Release Execution-Target Coverage Audit v1

## 为什么做这个审计

Reasoning-preservation SFT v4 在一组 fresh dev 上是 16/20 → 16/20，
`verified-reasoning` 是 0/4 → 0/4。此前 process-trace v6 把步骤格式从
28/32 提到 32/32，但最终答案本来就是 32/32，随后在 211-case matched
benchmark 上相对 base 4B 显著退化。

所以现在不能直接再训。先回答一个更具体的问题：现有训练数据有没有在同一条样本
里同时覆盖目标公式关系、正确中间执行和最终答案？

## 审计对象

- skill release 中当前实际选中的前 16 条 reasoning train；
- skill release 全部 3,964
  条 reasoning train；
- v6 的 160 条 process-trace train；
- v4 fresh dev 中 4 条错误的 `verified-reasoning` case。

## 关键结果

### Release reasoning 数据

- 16/16 都是 `FINAL: <number>`，显式过程监督 0/16；
- 目标关系覆盖：4/4；
- 同时具备过程监督的目标关系覆盖：
  0/4；
- 每条 prompt 中位数包含
  17 条无关 synthetic evidence；
- 目标答案泄露到 user prompt：
  0 条。

它覆盖了公式关系，但只监督最终 token，不监督怎么得到最终值。

### v6 process-trace 数据

- 显式过程监督：160/
  160；
- AST 形状覆盖：4/4；
- 相同“加数在最后再次被减掉”关系覆盖：
  0/4；
- v4 四个目标结果的 exact-result 覆盖：
  0/4。

它监督了步骤，但没有覆盖当前错误所需的重复操作数关系。

### 合并后

- shape 覆盖：4/4；
- relation 覆盖：4/4；
- process relation 覆盖：0/4；
- 同一条样本同时满足“目标 relation + process”的行数：
  0。

两套数据各覆盖一半，合起来仍然没有一条训练样本同时提供完整机制监督。

## 最小 Fresh Data Contract

下一步只允许生成数据，暂不允许训练：

- 512 train / 80 dev；
- 至少 300,000 train tokens；
- 128 个 fresh semantic task，覆盖 8×8 operand grid 和 multiplier 2/3；
- 每个 semantic task 同时生成 verified process view 和 final-only view；
- 256 条 JSON preservation replay，每个 JSON family 64 条；
- dev 中 24 个 fresh relation task、48 个 paired execution view、32 条
  JSON non-regression；
- train/dev 与旧 release semantic overlap 必须是 0；
- 所有 process view 必须逐步执行验证；
- 禁止 benchmark、independent holdout、模型输出和答案泄露进入数据。

数据合同全部通过之前，SFT、benchmark、holdout 和 RL 都保持关闭。

## 决策

- more_sft_allowed_now：false；
- generate_contract_dataset_next：true；
- reuse_answer_only_oversampling：false；
- reuse_process_v6_unchanged：false；
- benchmark / independent holdout / RL：全部关闭。

## Evidence

- accepted JSONL SHA256:
  `b5503761900cfa290dba03aff306ff511630d01544a4dab93de3a0be1e74abc1`;
- process v6 SHA256:
  `0e53fb3d05fb60569a4109da05b66d93c1158f734495e0126a55cf195c41653a`;
- failure manifest SHA256:
  `c55fee0eb358c3c3938b3b74ac713abe7253e4eac16091da59c65e07507ba3f1`;
- failure source commit: `ee82dab`.

## 结论边界

这次审计只比较 public-safe synthetic 训练机制，没有运行或评估模型。它不能证明模型能力提升，也不允许启动训练、访问 benchmark 或 independent holdout，或启动 RL。
