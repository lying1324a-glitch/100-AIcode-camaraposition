# ComfyUI 全景图尺寸换算节点

该仓库提供 4 个可串联的 ComfyUI 自定义节点，对应你给出的流程：

1. 全景图中找到一面墙（手工框选）
2. 测量墙像素宽
3. 用房间尺寸标定比例
4. 测量目标像素宽
5. 换算目标真实尺寸

## 节点列表

- `PanoramaWallCrop (BBox)`
  - 输入：`image`, `x1`, `y1`, `x2`, `y2`
  - 输出：`wall_image`, `wall_pixel_width`
  - 作用：框选墙体区域并输出墙像素宽度。 

- `PixelScaleFromRoomSize`
  - 输入：`wall_pixel_width`, `wall_real_width_m`
  - 输出：`meter_per_pixel`, `pixel_per_meter`
  - 作用：根据已知墙体真实宽度（米）计算比例尺。 

- `TargetPixelWidth (BBox)`
  - 输入：`image`, `target_x1`, `target_x2`
  - 输出：`target_pixel_width`
  - 作用：框选目标左右边界，计算目标像素宽。 

- `PixelToRealSize`
  - 输入：`target_pixel_width`, `meter_per_pixel`
  - 输出：`target_width_m`, `target_width_cm`
  - 作用：把像素宽度换算为真实宽度。 

## 典型工作流连接

`Load Image`
→ `PanoramaWallCrop`
→ `PixelScaleFromRoomSize`
→ `TargetPixelWidth`
→ `PixelToRealSize`

## 安装

将本仓库放到 ComfyUI 的 `custom_nodes` 目录下并重启 ComfyUI。

例如：

```bash
ComfyUI/custom_nodes/100-AIcode-camaraposition
```

## 精度提示

- 全景图存在透视畸变时，建议选择与目标同深度平面的墙面做标定。 
- 若目标不在标定墙同一平面，结果会有系统误差。 
- 可通过多点标定（不同墙段）扩展为分段比例模型。
