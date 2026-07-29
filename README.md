# Rocket Attitude Control / 火箭姿态控制

**Train a reinforcement-learning controller to rotate a simulated rigid rocket from about 35° to 120° using eight on/off attitude thrusters—while following a reference trajectory, respecting angular-rate limits, and reducing propellant use.**

**本项目训练一个强化学习控制器，通过八个开关式姿态推力器，将模拟火箭从约 35° 转动到 120°，同时跟踪参考轨迹、满足角速度约束并减少推进剂消耗。**

<p align="center">
  <img src="artifacts/demos/a2c_2460000_seed0_synchronized_preview.gif" width="100%" alt="Dynamic telemetry and rocket attitude interface">
</p>
<p align="center"><a href="artifacts/demos/a2c_2460000_seed0_dynamic_plot.mp4">Dynamic telemetry MP4 / 动态遥测 MP4</a>　·　<a href="artifacts/demos/a2c_2460000_seed0_rocket_ui.mp4">Rocket attitude MP4 / 火箭姿态 MP4</a></p>

The left panel compares the actual and target Euler angles (`gamma`, `psi`, `phi`), angular rates, thruster firings, and cumulative thrust impulse. The right panel shows three attitude projections and the eight-thruster layout. One 130-second simulation episode is compressed into a 10.9-second preview.

左侧展示实际/目标姿态、角速度、推力器点火和累计推力冲量；右侧展示三个姿态投影及八个推力器的布局。一次 130 秒的仿真被压缩为 10.9 秒预览。

[![CI](https://github.com/yusiwei77-star/rocket-attitude-control/actions/workflows/ci.yml/badge.svg)](https://github.com/yusiwei77-star/rocket-attitude-control/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this project does / 项目做什么

At every 0.1-second simulation step, an Advantage Actor-Critic (A2C) policy chooses which of eight 400 N attitude thrusters to fire. It observes only the three-axis attitude error and angular-rate error—not the full reference trajectory or future states.

在每个 0.1 秒仿真步，A2C 策略决定八个 400 N 姿态推力器中哪些点火。策略只观察三轴姿态误差和角速度误差，不直接读取完整参考轨迹或未来状态。

| Item / 项目 | Definition / 定义 |
|---|---|
| Plant / 被控对象 | Simplified three-axis rigid-body rotational dynamics / 简化的三轴刚体转动动力学 |
| Initial state / 初始状态 | Attitude near `[0°, 0°, 35°]` with seeded attitude and rate disturbances / 姿态约为 `[0°, 0°, 35°]`，并加入可复现的姿态与角速度扰动 |
| Observation / 观测 | `angle_error[3]` in radians, `angle_velocity_error[3]` in rad/s / 三轴姿态误差（rad）和角速度误差（rad/s） |
| Action / 动作 | `MultiBinary(8)`; each bit switches one 400 N thruster / 八位 0/1 动作，每位控制一个 400 N 推力器 |
| Episode / 回合 | 1,300 fixed steps, from 0 to exactly 130.0 s / 1,300 个固定步，精确结束于 130.0 秒 |
| Goal / 目标 | Track the nominal maneuver, finish near `[0°, 0°, 120°]`, satisfy rate limits, and minimize commanded thrust impulse / 跟踪标称机动、最终接近 `[0°, 0°, 120°]`、满足角速度限制并减少指令推力冲量 |

The repository includes the Gymnasium environment, training and evaluation commands, a pretrained checkpoint, deterministic regression tests, and synchronized video generation.

仓库包含 Gymnasium 环境、训练与评估命令、预训练权重、确定性回归测试和同步视频生成工具。

## Verified result / 已验证结果

The included checkpoint [`models/a2c_2460000.zip`](models/a2c_2460000.zip) was evaluated deterministically on seeds 0–19:

- At 70 s, absolute angular rates must be below `[0.5°, 1.0°, 1.0°]/s`.
- At 130 s, every attitude angle must be within 3° of `[0°, 0°, 120°]`, with the same angular-rate limits.
- Fuel impulse is the sum of commanded thrust over time (`N·s`), used here as a proxy for propellant consumption.

公开权重在 seeds 0–19 上进行了确定性评估：70 秒时三轴角速度必须低于 `[0.5°, 1.0°, 1.0°]/s`；130 秒时姿态必须落在目标值 ±3° 内并满足相同角速度约束。燃料冲量是指令推力随时间的积分，用作推进剂消耗的代理指标。

| Metric / 指标 | Seeds 0–19 |
|---|---:|
| 70 s rate-constraint success / 70 秒角速度约束成功率 | 100% |
| 130 s final success / 130 秒最终成功率 | 100% |
| Fuel impulse / 燃料冲量 | 23,838 ± 11,295 N·s |
| Attitude RMSE / 姿态 RMSE | 0.606° |
| Angular-rate RMSE / 角速度 RMSE | 0.044°/s |

The preview uses seed 0; its result is one sample rather than the 20-seed average. Reproduce the aggregate table with `rocket-evaluate`.

预览使用 seed 0，它只是一个样本，不代表 20 个种子的平均结果。可使用 `rocket-evaluate` 复现汇总表。

- [Seed-0 result summary / Seed-0 结果摘要](artifacts/results/a2c_2460000_seed0_summary.json)
- [Seed-0 compressed trajectory / Seed-0 压缩轨迹](artifacts/results/a2c_2460000_seed0_trajectory.npz)
- [20-seed regression test / 20-seed 回归测试](tests/test_policy.py)

## Quick start / 快速开始

Python 3.9 or newer is required. Python 3.11 is the CI-tested version. Clone the repository, create a virtual environment, and install the package.

需要 Python 3.9 或更高版本；CI 当前验证 Python 3.11。克隆仓库后创建虚拟环境并安装项目。

```bash
git clone https://github.com/yusiwei77-star/rocket-attitude-control.git
cd rocket-attitude-control
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the pretrained controller once and regenerate both MP4 files plus the combined GIF preview:

使用预训练控制器运行一次仿真，并重新生成两段 MP4 和合成 GIF：

```bash
rocket-render --model models/a2c_2460000.zip --seed 0 --device cpu
```

Outputs are written to `artifacts/demos/` and `artifacts/results/`. Evaluate all 20 fixed seeds with:

输出写入 `artifacts/demos/` 和 `artifacts/results/`。评估 20 个固定种子：

```bash
rocket-evaluate --model models/a2c_2460000.zip --episodes 20 --start-seed 0 --device cpu --output artifacts/results/evaluation.json
```

## How it works / 工作原理

1. `RocketSimulation` maps the eight thruster switches to body moments and integrates the nonlinear Euler-angle dynamics.
2. `RocketEnv` exposes the simulation as a Gymnasium dictionary observation and `MultiBinary(8)` action space.
3. The reward combines trajectory error, convergence direction, the 70/130-second constraints, and a penalty for every active thruster.
4. Stable-Baselines3 trains a `MultiInputPolicy` with A2C; evaluation uses deterministic actions.

1. `RocketSimulation` 将八个推力器开关转换为力矩，并积分非线性欧拉角动力学。
2. `RocketEnv` 将仿真封装为 Gymnasium 字典观测和 `MultiBinary(8)` 动作空间。
3. 奖励综合考虑轨迹误差、收敛方向、70/130 秒约束及推力器使用惩罚。
4. Stable-Baselines3 使用 A2C 训练 `MultiInputPolicy`，评估时采用确定性动作。

## Training / 训练

A short experiment defaults to 100,000 steps:

```bash
rocket-train --seed 0 --device cpu --output models/a2c_latest.zip
```

Use 2,460,000 steps to match the published checkpoint's training horizon:

```bash
rocket-train --timesteps 2460000 --seed 0 --device cpu --checkpoint-freq 100000 --output models/a2c_latest.zip
```

The checkpoint-compatible hyperparameters are fixed in the CLI: `n_steps=5`, learning rate `0.0007`, `gamma=0.99`, `gae_lambda=1.0`, `vf_coef=0.5`, and `max_grad_norm=0.5`. Exact learned weights can still vary across hardware and library builds.

CLI 固定使用与公开权重兼容的超参数；由于硬件、依赖和浮点计算差异，重新训练得到的具体参数仍可能不同。

## Scope and limitations / 适用范围与限制

This is a reproducible control and reinforcement-learning experiment, not flight-ready guidance, navigation, and control software. The model focuses on rotational motion and omits translation, gravity, aerodynamics, mass depletion, sensor noise, actuator delay, minimum impulse bit, and thruster failures. Fuel impulse is an optimization proxy rather than a propellant-mass model.

本项目是可复现的控制与强化学习实验，不是可直接用于飞行的制导、导航与控制软件。模型聚焦转动运动，未包含平动、重力、空气动力、质量变化、传感器噪声、执行器延迟、最小脉冲宽度和推力器故障；燃料冲量只是优化代理量，并非推进剂质量模型。

The automated workflow currently runs on Ubuntu with Python 3.11. The code is designed to be cross-platform and Windows commands are provided above, but Windows is not yet covered by CI.

自动化测试当前运行于 Ubuntu + Python 3.11。代码按跨平台方式实现并提供了 Windows 命令，但 Windows 尚未纳入 CI。

## Tests / 测试

```bash
pytest
```

The tests cover deterministic reset and termination, observation/action spaces, checkpoint loading, the 20-seed policy regression, MP4 metadata, and synchronized GIF metadata.

测试覆盖确定性重置与终止、观测/动作空间、权重加载、20-seed 策略回归、MP4 元数据及同步 GIF 元数据。

## Project layout / 项目结构

```text
src/rocket_attitude_control/
  simulation.py               # rigid-body dynamics, reward, constraints
  nominal.py                  # reference attitude trajectory
  env.py                      # Gymnasium adapter
  rendering.py                # pygame attitude renderer
  rollout.py                  # evaluation and trajectory collection
  video.py                    # MP4 and synchronized GIF generation
  cli/                        # rocket-train, rocket-evaluate, rocket-render
models/                       # published A2C checkpoint
artifacts/                    # demonstrations and seed-0 results
tests/                        # environment, policy, and media regression tests
```

## License / 许可

Released under the [MIT License](LICENSE). / 本项目采用 [MIT License](LICENSE)。
