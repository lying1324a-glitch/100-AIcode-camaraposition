import json
import math

import torch
import torch.nn.functional as F


ROOM_SHAPES = ["rectangle", "circle", "triangle"]


class PanoramaDistortionScaleTableNode:
    """
    节点1：
    - 输入：全景图 + 房间长宽高
    - 自动估计图像各水平位置的畸变特性（基于边缘方向一致性推断直线弧度趋势）
    - 按基础比例尺进行修正，输出“畸变特性-比例尺”表
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "room_length_m": ("FLOAT", {"default": 5.0, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "room_width_m": ("FLOAT", {"default": 4.0, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "room_height_m": ("FLOAT", {"default": 2.8, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "feature_bins": ("INT", {"default": 32, "min": 8, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("distortion_scale_table", "base_scale_m_per_px")
    FUNCTION = "build_table"
    CATEGORY = "Panorama/Distortion"

    def build_table(self, image, room_length_m, room_width_m, room_height_m, feature_bins):
        _validate_image(image)
        _, h, w, _ = image.shape

        base_scale = math.sqrt(float(room_length_m) ** 2 + float(room_width_m) ** 2) / float(w)

        feature_map = _compute_distortion_feature_map(image)
        feature_values = feature_map.flatten()

        # 使用房间高度作为修正强度的轻量先验；房间越高，垂向弧线可见性一般更强。
        height_factor = max(0.8, min(1.2, float(room_height_m) / 2.8))
        corrected_scale = base_scale * (1.0 + feature_values * 0.45 * height_factor)

        table = _make_feature_scale_table(feature_values, corrected_scale, int(feature_bins))
        payload = {
            "version": 1,
            "image_width": int(w),
            "image_height": int(h),
            "base_scale_m_per_px": float(base_scale),
            "room": {
                "length_m": float(room_length_m),
                "width_m": float(room_width_m),
                "height_m": float(room_height_m),
            },
            "feature_scale_table": table,
        }
        return (json.dumps(payload, ensure_ascii=False), float(base_scale))


class PanoramaDistortionFeatureNode:
    """
    节点2：
    - 输入：全景图
    - 自适应估计该图像的整体畸变特性值（球面投影场景）
    - 输出：用于匹配节点1表格的数值
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("distortion_feature_value",)
    FUNCTION = "estimate_feature"
    CATEGORY = "Panorama/Distortion"

    def estimate_feature(self, image):
        _validate_image(image)
        feature_map = _compute_distortion_feature_map(image)
        # 采用 70% 分位之前的稳健聚合：全局数值用中位+高位加权
        median = torch.quantile(feature_map, 0.5).item()
        q70 = torch.quantile(feature_map, 0.7).item()
        global_feature = 0.4 * median + 0.6 * q70
        return (float(global_feature),)


class DistortionScaleLookupNode:
    """
    节点3：
    - 输入：节点1输出的畸变特性-比例尺表 + 节点2输出的畸变特性值
    - 输出：匹配比例尺（取匹配集合的 70% 分位点）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "distortion_scale_table": ("STRING",),
                "distortion_feature_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "feature_tolerance": ("FLOAT", {"default": 0.06, "min": 0.001, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("matched_scale_m_per_px", "matched_sample_count")
    FUNCTION = "lookup_scale"
    CATEGORY = "Panorama/Distortion"

    def lookup_scale(self, distortion_scale_table, distortion_feature_value, feature_tolerance):
        data = json.loads(distortion_scale_table)
        rows = data.get("feature_scale_table", [])
        if not rows:
            raise ValueError("distortion_scale_table 中不存在 feature_scale_table 或为空")

        target = float(distortion_feature_value)
        tol = float(feature_tolerance)

        matched = [r for r in rows if abs(float(r["feature"]) - target) <= tol]

        # 若容差内无样本，回退到最近邻的若干样本，避免空匹配。
        if not matched:
            sorted_rows = sorted(rows, key=lambda r: abs(float(r["feature"]) - target))
            k = max(3, int(len(sorted_rows) * 0.1))
            matched = sorted_rows[:k]

        scales = torch.tensor([float(r["scale_m_per_px"]) for r in matched], dtype=torch.float32)
        q70_scale = torch.quantile(scales, 0.7).item()
        return (float(q70_scale), int(len(matched)))


class ScaledDimensionsNode:
    """
    节点4：
    - 输入：宽、高、比例尺
    - 兼容 ComfyUI “获取图像尺寸”节点输出的 width/height（INT）
    - 输出：按比例尺缩放后的宽高（width * scale, height * scale）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 0, "max": 100000, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 100000, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("scaled_width", "scaled_height")
    FUNCTION = "compute_scaled_dimensions"
    CATEGORY = "Panorama/Utility"

    def compute_scaled_dimensions(self, width, height, scale):
        scaled_width = float(width) * float(scale)
        scaled_height = float(height) * float(scale)
        return (scaled_width, scaled_height)


class ProportionalVolumeLimiterNode:
    """
    节点5：
    - 输入：长、宽、高、scaled_width、scaled_height
    - 按长宽高比例计算 4 种约束下的体积，并取最小体积方案
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "length": ("FLOAT", {"default": 5.0, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "width": ("FLOAT", {"default": 4.0, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "height": ("FLOAT", {"default": 2.8, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "scaled_width": ("FLOAT", {"default": 1.0, "min": 0.0001, "max": 100000.0, "step": 0.001}),
                "scaled_height": ("FLOAT", {"default": 1.0, "min": 0.0001, "max": 100000.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "length",
        "width",
        "height",
        "output",
        "volume_with_scaled_width_as_length",
        "volume_with_scaled_width_as_width",
        "volume_with_scaled_width_as_diagonal",
        "volume_with_scaled_height_as_height",
    )
    FUNCTION = "compute_min_volume"
    CATEGORY = "Panorama/Utility"

    def compute_min_volume(self, length, width, height, scaled_width, scaled_height):
        l = float(length)
        w = float(width)
        h = float(height)
        sw = float(scaled_width)
        sh = float(scaled_height)

        volume1, dims1 = self._volume_when_target_length(l, w, h, sw)
        volume2, dims2 = self._volume_when_target_width(l, w, h, sw)
        volume3, dims3 = self._volume_when_target_diagonal(l, w, h, sw)
        volume4, dims4 = self._volume_when_target_height(l, w, h, sh)

        candidates = [
            (volume1, dims1),
            (volume2, dims2),
            (volume3, dims3),
            (volume4, dims4),
        ]
        output_volume, selected_dims = min(candidates, key=lambda item: item[0])

        return (
            float(selected_dims[0]),
            float(selected_dims[1]),
            float(selected_dims[2]),
            float(output_volume),
            float(volume1),
            float(volume2),
            float(volume3),
            float(volume4),
        )

    @staticmethod
    def _volume_when_target_length(length, width, height, target_length):
        factor = target_length / length
        new_length = target_length
        new_width = width * factor
        new_height = height * factor
        volume = new_length * new_width * new_height
        return volume, (new_length, new_width, new_height)


    @staticmethod
    def _volume_when_target_width(length, width, height, target_width):
        factor = target_width / width
        new_length = length * factor
        new_width = target_width
        new_height = height * factor
        volume = new_length * new_width * new_height
        return volume, (new_length, new_width, new_height)

    @staticmethod
    def _volume_when_target_diagonal(length, width, height, target_diagonal):
        base_diagonal = math.sqrt(length * length + width * width)
        factor = target_diagonal / base_diagonal
        new_length = length * factor
        new_width = width * factor
        new_height = height * factor
        volume = new_length * new_width * new_height
        return volume, (new_length, new_width, new_height)

    @staticmethod
    def _volume_when_target_height(length, width, height, target_height):
        factor = target_height / height
        new_length = length * factor
        new_width = width * factor
        new_height = target_height
        volume = new_length * new_width * new_height
        return volume, (new_length, new_width, new_height)


class PanoramaDepthCropSizeEstimatorNode:
    """
    节点6：
    - 输入：全景深度图 + 房间长宽高 + 从全景中截出的局部图
    - 自动在深度图中做模板匹配定位截取区域
    - 基于球面全景几何关系估计该局部区域的实际宽高尺寸（米）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_image": ("IMAGE",),
                "room_length_m": ("FLOAT", {"default": 5.0, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "room_width_m": ("FLOAT", {"default": 4.0, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "room_height_m": ("FLOAT", {"default": 2.8, "min": 0.0001, "max": 10000.0, "step": 0.01}),
                "crop_image": ("IMAGE",),
                "depth_is_meters": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = (
        "estimated_width_m",
        "estimated_height_m",
        "estimated_area_m2",
        "estimated_depth_m",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "match_score",
    )
    FUNCTION = "estimate_size"
    CATEGORY = "Panorama/Depth"

    def estimate_size(self, depth_image, room_length_m, room_width_m, room_height_m, crop_image, depth_is_meters):
        _validate_image(depth_image)
        _validate_image(crop_image)

        depth_gray = _to_gray(depth_image)
        crop_gray = _to_gray(crop_image)

        depth_h, depth_w = depth_gray.shape[-2], depth_gray.shape[-1]
        crop_h, crop_w = crop_gray.shape[-2], crop_gray.shape[-1]
        if crop_h > depth_h or crop_w > depth_w:
            raise ValueError("crop_image 尺寸不能大于 depth_image")

        x, y, score = _find_best_template_match(depth_gray, crop_gray)

        patch = depth_gray[:, :, y : y + crop_h, x : x + crop_w]
        depth_m = _estimate_depth_in_meters(
            patch=patch,
            full_depth=depth_gray,
            room_length_m=float(room_length_m),
            room_width_m=float(room_width_m),
            room_height_m=float(room_height_m),
            depth_is_meters=bool(depth_is_meters),
        )

        yaw_per_px = (2.0 * math.pi) / float(depth_w)
        pitch_per_px = math.pi / float(depth_h)

        width_m = float(depth_m) * float(crop_w) * yaw_per_px
        height_m = float(depth_m) * float(crop_h) * pitch_per_px
        area_m2 = width_m * height_m

        return (
            float(width_m),
            float(height_m),
            float(area_m2),
            float(depth_m),
            int(x),
            int(y),
            int(crop_w),
            int(crop_h),
            float(score),
        )


class PanoramaDepthFusionFromStitchNode:
    """
    节点7：
    - 输入：多张深度图（batch）+ 全景图合成结果
    - 假设多张深度图已经通过全景流程对齐到近似一致的观察方向
    - 输出：融合后的大幅全景深度图（与全景图同分辨率）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_images": ("IMAGE",),
                "panorama_image": ("IMAGE",),
                "depth_is_meters": ("BOOLEAN", {"default": False}),
                "blend_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("fused_depth_image", "depth_scale_factor", "source_count")
    FUNCTION = "fuse_depth"
    CATEGORY = "Panorama/Depth"

    def fuse_depth(self, depth_images, panorama_image, depth_is_meters, blend_strength):
        _validate_image(depth_images)
        _validate_image(panorama_image)

        pano_h = panorama_image.shape[1]
        pano_w = panorama_image.shape[2]
        source_count = int(depth_images.shape[0])

        gray_depth = _to_gray_batch(depth_images)
        resized = F.interpolate(gray_depth, size=(pano_h, pano_w), mode="bilinear", align_corners=False)

        # 使用全景亮度梯度抑制接缝处的强噪声，提升融合稳定性。
        pano_gray = _to_gray(panorama_image)
        gx = pano_gray[:, :, :, 1:] - pano_gray[:, :, :, :-1]
        gy = pano_gray[:, :, 1:, :] - pano_gray[:, :, :-1, :]
        grad = torch.zeros_like(pano_gray)
        grad[:, :, :, 1:] += torch.abs(gx)
        grad[:, :, 1:, :] += torch.abs(gy)
        edge_weight = 1.0 / (1.0 + grad)

        per_map_conf = _compute_depth_confidence(resized)
        per_map_conf = per_map_conf * edge_weight

        alpha = float(blend_strength)
        median_map = torch.median(resized, dim=0, keepdim=True).values
        weighted_mean = torch.sum(resized * per_map_conf, dim=0, keepdim=True) / (torch.sum(per_map_conf, dim=0, keepdim=True) + 1e-8)
        fused = alpha * weighted_mean + (1.0 - alpha) * median_map

        if not bool(depth_is_meters):
            q95 = float(torch.quantile(fused.view(-1), 0.95).item())
            scale = 1.0 / max(q95, 1e-8)
            fused = fused * scale
        else:
            scale = 1.0

        fused = fused.clamp(min=0.0)
        fused_rgb = fused.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        return (fused_rgb, float(scale), int(source_count))


class RoomPoseFromPanoramaNode:
    """
    节点8：
    - 输入：全景图 + 融合深度图 + 房间形状与尺寸
    - 输出：估计相机位姿（位置+欧拉角）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama_image": ("IMAGE",),
                "fused_depth_image": ("IMAGE",),
                "room_shape": (ROOM_SHAPES,),
                "room_size_a_m": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 1000.0, "step": 0.01}),
                "room_size_b_m": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 1000.0, "step": 0.01}),
                "room_height_m": ("FLOAT", {"default": 2.8, "min": 0.1, "max": 1000.0, "step": 0.01}),
                "camera_height_prior_m": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("camera_pose_json", "camera_x_m", "camera_y_m", "camera_z_m", "yaw_deg", "pitch_deg", "roll_deg")
    FUNCTION = "estimate_pose"
    CATEGORY = "Panorama/Pose"

    def estimate_pose(self, panorama_image, fused_depth_image, room_shape, room_size_a_m, room_size_b_m, room_height_m, camera_height_prior_m):
        _validate_image(panorama_image)
        _validate_image(fused_depth_image)

        pano_gray = _to_gray(panorama_image)
        depth = _to_gray(fused_depth_image)
        _, _, h, w = pano_gray.shape

        yaw_deg = _estimate_yaw_from_panorama(pano_gray)
        pitch_deg = _estimate_pitch_from_panorama(pano_gray)
        roll_deg = 0.0

        left = torch.median(depth[:, :, :, : w // 4]).item()
        right = torch.median(depth[:, :, :, 3 * w // 4 :]).item()
        front = torch.median(depth[:, :, :, w // 4 : w // 2]).item()
        back = torch.median(depth[:, :, :, w // 2 : 3 * w // 4]).item()

        size_a = float(room_size_a_m)
        size_b = float(room_size_b_m)

        if room_shape == "circle":
            radius = max(size_a * 0.5, 1e-4)
            x_m = float((right - left) * 0.15)
            y_m = float((back - front) * 0.15)
            norm = math.sqrt(x_m * x_m + y_m * y_m)
            if norm > radius * 0.95:
                scale = (radius * 0.95) / max(norm, 1e-8)
                x_m *= scale
                y_m *= scale
        elif room_shape == "triangle":
            x_m = float((right - left) * 0.12)
            y_m = float((back - front) * 0.12)
            x_m = max(-size_a * 0.45, min(size_a * 0.45, x_m))
            y_m = max(-size_b * 0.3, min(size_b * 0.3, y_m))
        else:
            x_m = float((right - left) * 0.2)
            y_m = float((back - front) * 0.2)
            x_m = max(-size_a * 0.5, min(size_a * 0.5, x_m))
            y_m = max(-size_b * 0.5, min(size_b * 0.5, y_m))

        top_depth = torch.median(depth[:, :, : max(1, h // 6), :]).item()
        bottom_depth = torch.median(depth[:, :, -max(1, h // 6) :, :]).item()
        z_adjust = (bottom_depth - top_depth) * 0.05
        z_m = max(0.1, min(float(room_height_m) - 0.1, float(camera_height_prior_m) + float(z_adjust)))

        pose = {
            "version": 1,
            "room_shape": room_shape,
            "room_size_a_m": size_a,
            "room_size_b_m": size_b,
            "room_height_m": float(room_height_m),
            "camera_position_m": {"x": float(x_m), "y": float(y_m), "z": float(z_m)},
            "camera_rotation_deg": {"yaw": float(yaw_deg), "pitch": float(pitch_deg), "roll": float(roll_deg)},
        }
        return (json.dumps(pose, ensure_ascii=False), float(x_m), float(y_m), float(z_m), float(yaw_deg), float(pitch_deg), float(roll_deg))


class PanoramaCropMetricEstimatorNode:
    """
    节点9：
    - 输入：融合深度图 + 相机位姿 +（框坐标 / crop_image模板匹配 / boxes_json）
    - 输出：该区域的实际宽高/面积估计
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fused_depth_image": ("IMAGE",),
                "camera_pose_json": ("STRING",),
                "crop_x": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "crop_y": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "crop_w": ("INT", {"default": 256, "min": 1, "max": 100000, "step": 1}),
                "crop_h": ("INT", {"default": 256, "min": 1, "max": 100000, "step": 1}),
                "depth_is_meters": ("BOOLEAN", {"default": False}),
                "box_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                "panorama_image": ("IMAGE",),
                "crop_image": ("IMAGE",),
                "boxes_json": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "estimated_width_m",
        "estimated_height_m",
        "estimated_area_m2",
        "median_depth_m",
        "center_yaw_deg",
        "center_pitch_deg",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "match_score",
        "bbox_source",
    )
    FUNCTION = "estimate_crop_metric"
    CATEGORY = "Panorama/Measurement"

    def estimate_crop_metric(
        self,
        fused_depth_image,
        camera_pose_json,
        crop_x,
        crop_y,
        crop_w,
        crop_h,
        depth_is_meters,
        box_index,
        panorama_image=None,
        crop_image=None,
        boxes_json="",
    ):
        _validate_image(fused_depth_image)
        pose = json.loads(camera_pose_json)

        depth = _to_gray(fused_depth_image)
        _, _, pano_h, pano_w = depth.shape

        x0 = int(crop_x)
        y0 = int(crop_y)
        w = int(crop_w)
        h = int(crop_h)
        match_score = 1.0
        bbox_source = "manual"

        # 优先级1：boxes_json + box_index
        parsed_box = _extract_box_from_json(boxes_json, int(box_index))
        if parsed_box is not None:
            x0, y0, w, h = parsed_box
            bbox_source = "boxes_json"

        # 优先级2：crop_image + panorama_image 模板匹配（覆盖 manual，低于 boxes_json）
        elif crop_image is not None:
            if panorama_image is None:
                raise ValueError("当输入 crop_image 时，需同时输入 panorama_image 以定位裁剪区域")
            _validate_image(crop_image)
            _validate_image(panorama_image)

            pano_gray = _to_gray(panorama_image)
            crop_gray = _to_gray(crop_image)

            if crop_gray.shape[-2] > pano_gray.shape[-2] or crop_gray.shape[-1] > pano_gray.shape[-1]:
                raise ValueError("crop_image 尺寸不能大于 panorama_image")

            x0, y0, match_score = _find_best_template_match(pano_gray, crop_gray)
            w = int(crop_gray.shape[-1])
            h = int(crop_gray.shape[-2])
            bbox_source = "crop_image_match"

        x0 = int(max(0, min(x0, pano_w - 1)))
        y0 = int(max(0, min(y0, pano_h - 1)))
        w = int(max(1, min(w, pano_w - x0)))
        h = int(max(1, min(h, pano_h - y0)))

        patch = depth[:, :, y0 : y0 + h, x0 : x0 + w]
        depth_med = float(torch.median(patch).item())

        if bool(depth_is_meters):
            depth_m = max(depth_med, 1e-4)
        else:
            ref_a = float(pose.get("room_size_a_m", 5.0))
            ref_b = float(pose.get("room_size_b_m", 4.0))
            diag = math.sqrt(ref_a * ref_a + ref_b * ref_b)
            q95 = float(torch.quantile(depth.view(-1), 0.95).item())
            depth_m = max(1e-4, depth_med / max(q95, 1e-8) * diag)

        yaw_per_px = (2.0 * math.pi) / float(pano_w)
        pitch_per_px = math.pi / float(pano_h)

        cx = x0 + 0.5 * w
        cy = y0 + 0.5 * h
        center_yaw = (cx / float(pano_w)) * 2.0 * math.pi - math.pi
        center_pitch = 0.5 * math.pi - (cy / float(pano_h)) * math.pi

        width_m = depth_m * (w * yaw_per_px) * max(0.15, math.cos(center_pitch))
        height_m = depth_m * (h * pitch_per_px)
        area_m2 = width_m * height_m

        yaw_bias = float(pose.get("camera_rotation_deg", {}).get("yaw", 0.0))
        pitch_bias = float(pose.get("camera_rotation_deg", {}).get("pitch", 0.0))

        return (
            float(width_m),
            float(height_m),
            float(area_m2),
            float(depth_m),
            float(math.degrees(center_yaw) + yaw_bias),
            float(math.degrees(center_pitch) + pitch_bias),
            int(x0),
            int(y0),
            int(w),
            int(h),
            float(match_score),
            str(bbox_source),
        )



def _extract_box_from_json(boxes_json, box_index):
    if boxes_json is None:
        return None
    raw = str(boxes_json).strip()
    if not raw:
        return None

    data = json.loads(raw)
    if isinstance(data, dict):
        if "boxes" in data:
            boxes = data.get("boxes", [])
        elif all(k in data for k in ("x", "y", "w", "h")):
            boxes = [data]
        else:
            boxes = []
    elif isinstance(data, list):
        boxes = data
    else:
        boxes = []

    if not boxes:
        return None

    idx = max(0, min(int(box_index), len(boxes) - 1))
    item = boxes[idx]
    if not isinstance(item, dict):
        return None

    x = item.get("x", item.get("left", item.get("xmin", 0)))
    y = item.get("y", item.get("top", item.get("ymin", 0)))
    w = item.get("w", item.get("width", None))
    h = item.get("h", item.get("height", None))

    if w is None and "xmax" in item:
        w = float(item["xmax"]) - float(x)
    if h is None and "ymax" in item:
        h = float(item["ymax"]) - float(y)

    if w is None or h is None:
        return None

    return int(float(x)), int(float(y)), int(max(1.0, float(w))), int(max(1.0, float(h)))


def _validate_image(image):
    if image.ndim != 4:
        raise ValueError("Expected IMAGE tensor in shape [B, H, W, C].")
    if image.shape[-1] < 3:
        raise ValueError("Expected IMAGE with at least 3 channels.")


def _to_gray(image):
    # 仅使用第一张图，避免批处理干扰节点定义
    rgb = image[0, :, :, :3].float()
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    return gray.unsqueeze(0).unsqueeze(0)


def _to_gray_batch(image):
    rgb = image[:, :, :, :3].float()
    gray = 0.2989 * rgb[:, :, :, 0] + 0.5870 * rgb[:, :, :, 1] + 0.1140 * rgb[:, :, :, 2]
    return gray.unsqueeze(1)


def _compute_depth_confidence(depth_batch):
    # depth_batch: [N,1,H,W]
    dx = torch.abs(depth_batch[:, :, :, 1:] - depth_batch[:, :, :, :-1])
    dy = torch.abs(depth_batch[:, :, 1:, :] - depth_batch[:, :, :-1, :])

    grad = torch.zeros_like(depth_batch)
    grad[:, :, :, 1:] += dx
    grad[:, :, 1:, :] += dy
    conf = 1.0 / (1.0 + grad)
    return conf


def _estimate_yaw_from_panorama(pano_gray):
    # 基于列强度重心估计朝向偏移
    column_energy = torch.mean(pano_gray, dim=2).view(-1)
    idx = int(torch.argmax(column_energy).item())
    w = column_energy.shape[0]
    yaw = (float(idx) / float(w)) * 360.0 - 180.0
    return yaw


def _estimate_pitch_from_panorama(pano_gray):
    row_energy = torch.mean(pano_gray, dim=3).view(-1)
    idx = int(torch.argmax(row_energy).item())
    h = row_energy.shape[0]
    pitch = 90.0 - (float(idx) / float(h)) * 180.0
    return max(-45.0, min(45.0, pitch * 0.35))


def _compute_distortion_feature_map(image):
    """
    通过边缘方向在列方向上的不稳定性估计“直线弧度趋势”：
    - 梯度方向越混乱，代表局部直线更可能呈现弧线化（畸变更明显）
    - 输出按列的 [0,1] 特征图，供节点1/2共用
    """
    gray = _to_gray(image)

    sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], dtype=gray.dtype, device=gray.device)
    sobel_y = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], dtype=gray.dtype, device=gray.device)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)

    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    angle = torch.atan2(gy, gx)  # [-pi, pi]

    # 双角映射处理方向周期性（theta 与 theta+pi 等价）
    cos2 = torch.cos(2.0 * angle)
    sin2 = torch.sin(2.0 * angle)

    weight = mag / (torch.mean(mag) + 1e-12)

    # 沿高度聚合每一列的方向一致性，1-R 作为弧度/畸变特征
    c = torch.sum(weight * cos2, dim=2)
    s = torch.sum(weight * sin2, dim=2)
    wsum = torch.sum(weight, dim=2) + 1e-12

    resultant = torch.sqrt(c * c + s * s) / wsum
    feature = 1.0 - resultant.squeeze(0).squeeze(0)

    # 平滑，增强稳健性
    kernel = torch.ones((1, 1, 1, 9), dtype=feature.dtype, device=feature.device) / 9.0
    feature = F.conv2d(feature.view(1, 1, 1, -1), kernel, padding=(0, 4)).view(-1)

    fmin = torch.min(feature)
    fmax = torch.max(feature)
    norm = (feature - fmin) / (fmax - fmin + 1e-12)
    return norm.clamp(0.0, 1.0)


def _make_feature_scale_table(feature_values, scale_values, bins):
    if bins <= 0:
        raise ValueError("feature_bins must be positive")

    bin_edges = torch.linspace(0.0, 1.0, steps=bins + 1, device=feature_values.device)
    rows = []

    for i in range(bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i < bins - 1:
            mask = (feature_values >= lo) & (feature_values < hi)
        else:
            mask = (feature_values >= lo) & (feature_values <= hi)

        if torch.any(mask):
            fs = feature_values[mask]
            ss = scale_values[mask]
            rows.append(
                {
                    "feature": float(torch.mean(fs).item()),
                    "scale_m_per_px": float(torch.mean(ss).item()),
                    "sample_count": int(mask.sum().item()),
                }
            )

    if not rows:
        raise ValueError("无法构建畸变特性表：图像特征为空")
    return rows


def _find_best_template_match(search_image, template_image):
    """
    在单通道图像中寻找 template 最佳匹配位置，返回左上角与相似度分数。
    两者输入 shape: [1,1,H,W]
    """
    eps = 1e-12

    template = template_image - torch.mean(template_image)
    tnorm = torch.sqrt(torch.sum(template * template) + eps)

    kernel = template
    numerator = F.conv2d(search_image, kernel)

    ones = torch.ones_like(template)
    local_mean = F.conv2d(search_image, ones) / float(template.shape[-2] * template.shape[-1])
    centered = search_image - F.pad(
        local_mean,
        (
            template.shape[-1] // 2,
            template.shape[-1] - 1 - template.shape[-1] // 2,
            template.shape[-2] // 2,
            template.shape[-2] - 1 - template.shape[-2] // 2,
        ),
        mode="replicate",
    )
    local_energy = F.conv2d(centered * centered, ones)
    denom = torch.sqrt(local_energy + eps) * tnorm
    zncc = numerator / (denom + eps)

    flat_idx = int(torch.argmax(zncc).item())
    out_h, out_w = zncc.shape[-2], zncc.shape[-1]
    y = flat_idx // out_w
    x = flat_idx % out_w
    score = float(zncc[0, 0, y, x].item())
    return x, y, score


def _estimate_depth_in_meters(patch, full_depth, room_length_m, room_width_m, room_height_m, depth_is_meters):
    patch_median = float(torch.median(patch).item())
    if depth_is_meters:
        return max(1e-4, patch_median)

    # 对未标定深度图进行简单尺度归一：将95分位映射到房间平面对角线。
    # 该假设适配典型矩形房间全景：最远可见点通常接近水平对角尺度。
    room_diag = math.sqrt(room_length_m * room_length_m + room_width_m * room_width_m)
    q95 = float(torch.quantile(full_depth.view(-1), 0.95).item())
    if q95 <= 1e-8:
        raise ValueError("depth_image 数值无效，无法估计深度尺度")

    depth_m = patch_median / q95 * room_diag
    return max(1e-4, depth_m)


NODE_CLASS_MAPPINGS = {
    "PanoramaDistortionScaleTable": PanoramaDistortionScaleTableNode,
    "PanoramaDistortionFeature": PanoramaDistortionFeatureNode,
    "DistortionScaleLookup": DistortionScaleLookupNode,
    "ScaledDimensions": ScaledDimensionsNode,
    "ProportionalVolumeLimiter": ProportionalVolumeLimiterNode,
    "PanoramaDepthCropSizeEstimator": PanoramaDepthCropSizeEstimatorNode,
    "PanoramaDepthFusionFromStitch": PanoramaDepthFusionFromStitchNode,
    "RoomPoseFromPanorama": RoomPoseFromPanoramaNode,
    "PanoramaCropMetricEstimator": PanoramaCropMetricEstimatorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoramaDistortionScaleTable": "Panorama Distortion Scale Table",
    "PanoramaDistortionFeature": "Panorama Distortion Feature",
    "DistortionScaleLookup": "Distortion Scale Lookup (Q70)",
    "ScaledDimensions": "Scaled Dimensions",
    "ProportionalVolumeLimiter": "Proportional Volume Limiter",
    "PanoramaDepthCropSizeEstimator": "Panorama Depth Crop Size Estimator",
    "PanoramaDepthFusionFromStitch": "Panorama Depth Fusion (From Stitch Result)",
    "RoomPoseFromPanorama": "Room-constrained Camera Pose From Panorama",
    "PanoramaCropMetricEstimator": "Panorama Crop Real-world Size Estimator",
}
