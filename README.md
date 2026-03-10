# ComfyUI 全景图畸变比例尺节点

本仓库提供 **4 个节点**，用于球面投影全景图中的“畸变感知比例尺估计”与尺寸缩放计算。

## 设计逻辑

### 1) `Panorama Distortion Scale Table`
输入：`image`, `room_length_m`, `room_width_m`, `room_height_m`

功能：
- 自动分析图像各水平位置的畸变特性（基于边缘方向一致性，近似对应“直线弧度趋势”）。
- 基础比例尺按要求定义为：
  - `base_scale = sqrt(length^2 + width^2) / image_width`
- 用畸变特性对基础比例尺进行修正，输出“畸变特性-比例尺”表。

输出：
- `distortion_scale_table`（JSON 字符串）
- `base_scale_m_per_px`

---

### 2) `Panorama Distortion Feature`
输入：`image`

功能：
- 自适应估计图像整体畸变程度（球面投影前提）。

输出：
- `distortion_feature_value`（数值）

---

### 3) `Distortion Scale Lookup (Q70)`
输入：`distortion_scale_table`, `distortion_feature_value`

功能：
- 根据节点2给出的畸变值在节点1表格中匹配对应比例尺集合。
- 输出比例尺采用 **70% 分位点**（Q70）。

输出：
- `matched_scale_m_per_px`
- `matched_sample_count`

---

### 4) `Scaled Dimensions`
输入：`width(INT)`, `height(INT)`, `scale(FLOAT)`

功能：
- 计算缩放后的尺寸：
  - `scaled_width = width * scale`
  - `scaled_height = height * scale`
- 输出类型为 `FLOAT`，支持小数结果（例如 `0.12382`）。
- 适配 ComfyUI 自带“获取图像尺寸”节点输出的宽高整型数值。

输出：
- `scaled_width`
- `scaled_height`

## 典型工作流

`Load Image`
→ `Panorama Distortion Scale Table`
→ `Panorama Distortion Feature`
→ `Distortion Scale Lookup (Q70)`
→ `Scaled Dimensions`

## 安装

将本仓库放到 ComfyUI 的 `custom_nodes` 目录并重启 ComfyUI。

```bash
ComfyUI/custom_nodes/100-AIcode-camaraposition
```
