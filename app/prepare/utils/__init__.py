# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Task2 eval pipeline 的自包含工具库。

所有常量/关节名/姿态/控制逻辑都在本包内就地定义,**不 import** 仓库其它模块
(bridge_core / data_collect_prepare / base_holder),因此整个 `prepare/` 目录
可原样拷到相同 pixi 环境的另一台机子直接用。
"""
