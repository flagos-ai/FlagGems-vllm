---
title: 快速开始
weight: 10
bookCollapseSection: true
---

# FlagGems-vllm 快速开始

本节介绍 FlagGems-vllm 的运行环境依赖与安装部署。

## 前置环境要求

- **Python**: Python ≥ 3.8（推荐 3.11 或 3.12）
- **PyTorch**: PyTorch ≥ 2.6.0
- **FlagGems**: 必需核心依赖，通过 `pip install flag-gems==5.4.0` 安装（部分特化算子底层直接调用 FlagGems 通用算子）
- **编译器**: Triton（≥ 3.0）或 FlagTree 编译器
- **硬件支持**: 支持 CUDA 的 GPU（NVIDIA 平台）或其它支持的异构加速硬件（昇腾、AMD、寒武纪、海光等）
- **vLLM**: 可选依赖，用于端到端大模型推理与插件集成

## 安装部署

FlagGems-vllm 支持源码安装、可编辑开发安装、自动化环境搭建脚本以及与 FlagGems / vllm-plugin-fl 的全栈集成安装。

详细步骤请参阅[安装指南](install.md)。

## 下一步

- [安装指南](install.md) - 环境依赖准备与多种安装方式
- [用法指南](../usage/basic.md) - 算子调用、全局调度与常用 API
- [与 vllm-plugin-fl 集成](../usage/vllm-plugin.md) - vLLM 插件层协同
