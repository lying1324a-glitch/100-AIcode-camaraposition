# ComfyUI 全景/深度/位姿/尺寸估计节点

本仓库提供 **多节点工具集**，覆盖你提出的完整链路：
1. 多张深度图按全景合成结果融合为一张大深度图。
2. 已知规则房间形状与尺寸，反推相机位姿。
3. 用位姿 + 深度，对全景任意裁剪区域做真实尺寸估计。

## 关键新增节点（对应你的 3 个需求）

### A) `Panorama Depth Fusion (From Stitch Result)`
- 内部类名：`PanoramaDepthFusionFromStitch`
- 输入：`depth_images`（可为 batch 多张深度图）、`panorama_image`、`depth_is_meters`、`blend_strength`
- 输出：`fused_depth_image`、`depth_scale_factor`、`source_count`
- 作用：将多张深度图重采样到全景尺寸后，按置信度加权 + 中值稳健融合，得到和全景结果一致的大深度图。

### B) `Room-constrained Camera Pose From Panorama`
- 内部类名：`RoomPoseFromPanorama`
- 输入：`panorama_image`、`fused_depth_image`、`room_shape`（`rectangle/circle/triangle`）、`room_size_a_m`、`room_size_b_m`、`room_height_m`、`camera_height_prior_m`
- 输出：`camera_pose_json` + `camera_x/y/z_m` + `yaw/pitch/roll_deg`
- 作用：在“规则房间 + 已知尺寸”约束下给出可用的相机位姿估计。

### C) `Panorama Crop Real-world Size Estimator`
- 内部类名：`PanoramaCropMetricEstimator`
- 输入：
  - 必填：`fused_depth_image`、`camera_pose_json`、`crop_x/y/w/h`、`depth_is_meters`、`box_index`、`match_max_side`（字符串，留空自动用 1024）
  - 可选1：`panorama_image` + `crop_image`（自动模板匹配定位）
  - 可选2：`boxes_json`（直接读取框坐标，再按 `box_index` 选框）
- 输出：`estimated_width_m`、`estimated_height_m`、`estimated_area_m2`、`median_depth_m`、`center_yaw_deg`、`center_pitch_deg`、`bbox_x/y/w/h`、`match_score`、`bbox_source`
- 作用：对全景中任意裁剪区域做米制尺寸估计，并支持“传 crop 图自动找位置”或“传 boxes_json 直接取框”。`match_max_side`（字符串，留空自动用 1024） 可控制模板匹配降采样上限，避免超大全景时节点卡顿。

---

## 其他现有节点

1. `Image Tensor -> Numpy Bridge`（任意图像输入转 numpy 输出，并返回 source_device）
2. `Image Numpy -> Tensor Bridge (Device-aware)`（可按 source_device 或指定设备恢复为 torch tensor，便于后续继续在 GPU 跑）
3. `Panorama Distortion Scale Table`
4. `Panorama Distortion Feature`
5. `Distortion Scale Lookup (Q70)`
6. `Scaled Dimensions`
7. `Proportional Volume Limiter`
8. `Panorama Depth Crop Size Estimator`

这些节点仍可用于你原有的比例尺估计、体积约束和局部深度尺寸估计工作流。

## 推荐工作流（对应你的需求）

`Load Depth Images (batch)` + `Load Panorama Image`
→ `Panorama Depth Fusion (From Stitch Result)`
→ `Room-constrained Camera Pose From Panorama`
→ `Panorama Crop Real-world Size Estimator`


## Numpy-only 第三方节点的标准桥接工作流（GPU可恢复）

推荐直接按下面连接：

`torch节点`
→ `Image Tensor -> Numpy Bridge`
→ `(numpy-only第三方节点)`
→ `Image Numpy -> Tensor Bridge (Device-aware)`（`device_mode=source`，`source_device` 接前一节点输出）
→ `后续torch节点`

说明：
- `Image Tensor -> Numpy Bridge` 会输出 `numpy_image` 和 `source_device`（例如 `cuda:0`）。
- 将 `source_device` 连到 `Image Numpy -> Tensor Bridge (Device-aware)` 的同名输入，并把 `device_mode` 设为 `source`，即可在 numpy-only 节点之后自动回到原始 GPU 设备（若可用）。
- 若当前环境无 CUDA，节点会自动回退到 CPU，避免报错。

## 安装

将本仓库放到 ComfyUI 的 `custom_nodes` 目录并重启 ComfyUI：

```bash
ComfyUI/custom_nodes/100-AIcode-camaraposition
```
