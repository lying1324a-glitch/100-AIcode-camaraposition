# ComfyUI 全景/深度/位姿/尺寸估计节点

本仓库提供 **9 个节点**，覆盖你提出的完整链路：
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
- 输入：`fused_depth_image`、`camera_pose_json`、`crop_x/y/w/h`、`depth_is_meters`
- 输出：`estimated_width_m`、`estimated_height_m`、`estimated_area_m2`、`median_depth_m`、`center_yaw_deg`、`center_pitch_deg`
- 作用：对全景中任意裁剪区域做米制尺寸估计。

---

## 其他现有节点

1. `Panorama Distortion Scale Table`
2. `Panorama Distortion Feature`
3. `Distortion Scale Lookup (Q70)`
4. `Scaled Dimensions`
5. `Proportional Volume Limiter`
6. `Panorama Depth Crop Size Estimator`

这些节点仍可用于你原有的比例尺估计、体积约束和局部深度尺寸估计工作流。

## 推荐工作流（对应你的需求）

`Load Depth Images (batch)` + `Load Panorama Image`
→ `Panorama Depth Fusion (From Stitch Result)`
→ `Room-constrained Camera Pose From Panorama`
→ `Panorama Crop Real-world Size Estimator`

## 安装

将本仓库放到 ComfyUI 的 `custom_nodes` 目录并重启 ComfyUI：

```bash
ComfyUI/custom_nodes/100-AIcode-camaraposition
```
