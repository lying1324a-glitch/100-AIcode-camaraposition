import json
import math

import torch
import torch.nn.functional as F


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


NODE_CLASS_MAPPINGS = {
    "PanoramaDistortionScaleTable": PanoramaDistortionScaleTableNode,
    "PanoramaDistortionFeature": PanoramaDistortionFeatureNode,
    "DistortionScaleLookup": DistortionScaleLookupNode,
    "ScaledDimensions": ScaledDimensionsNode,
    "ProportionalVolumeLimiter": ProportionalVolumeLimiterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoramaDistortionScaleTable": "Panorama Distortion Scale Table",
    "PanoramaDistortionFeature": "Panorama Distortion Feature",
    "DistortionScaleLookup": "Distortion Scale Lookup (Q70)",
    "ScaledDimensions": "Scaled Dimensions",
    "ProportionalVolumeLimiter": "Proportional Volume Limiter",
}
