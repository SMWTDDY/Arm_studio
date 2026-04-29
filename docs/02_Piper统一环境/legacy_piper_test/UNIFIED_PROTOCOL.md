# 统一 Piper 协议

注意：`Piper_test` 现在主要作为历史备份目录。合并后的主工作目录是：

```text
Arm_studio/
```

最新协议说明请优先查看：

```text
Arm_studio/agent_infra/Piper_Env/UNIFIED_PROTOCOL.md
Arm_studio/MERGE_STATUS.md
```

以下内容保留为同步副本，避免旧目录文档和主目录状态不一致。

## 环境接口

所有统一 Piper 环境都应暴露：

```text
env.unwrapped.meta_keys
env.reset() -> obs, info
env.step(action_dict) -> obs, reward, terminated, truncated, info
env.unwrapped.get_safe_action() -> action_dict
```

标准观测结构：

```text
obs/state/<state_key>              float32 数组
obs/under_control/<arm_name>       bool 数组，shape 为 (1,)
obs/rgb/<camera_role>              uint8 CHW 图像
obs/depth/<camera_role>            可选 depth 图像
```

标准动作结构：

```text
action/<prefix>arm                 6D joint 或 pose 动作
action/<prefix>gripper             1D 夹爪动作
```

单臂无前缀，双臂使用 `left_` 和 `right_` 前缀。

## 当前已实现

- 真实单臂：`backend=real, arm_mode=single`
- 真实双臂：`backend=real, arm_mode=dual`
- 仿真单臂：`backend=sim, arm_mode=single`
- 仿真双臂协议入口：`backend=sim, arm_mode=dual`

当前双臂仿真由两个 Arm_studio 单臂 ManiSkill 场景组合而成，已经对齐真实双臂 key，但还不是同一物理场景内的双臂交互。

## 尚未完成

1. 真正共享场景的双臂 ManiSkill 环境。
2. 录制、回放、采集脚本统一读取 `unified_*.yaml`。
3. 旧 Arm_studio HDF5 转统一 raw H5 协议。
4. 真实/仿真双臂 H5 round-trip 实测。
5. 训练部分接入统一协议。
