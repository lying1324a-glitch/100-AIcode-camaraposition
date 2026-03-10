class PanoramaWallCropNode:
    """
    Step 1+2: from a panorama image, crop wall region and output wall pixel width.
    The wall is defined by manual bbox for reproducibility in ComfyUI workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x1": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "y1": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "x2": ("INT", {"default": 100, "min": 1, "max": 100000, "step": 1}),
                "y2": ("INT", {"default": 100, "min": 1, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("wall_image", "wall_pixel_width")
    FUNCTION = "crop_wall"
    CATEGORY = "Panorama/Measurement"

    def crop_wall(self, image, x1, y1, x2, y2):
        # ComfyUI IMAGE: [B, H, W, C]
        if image.ndim != 4:
            raise ValueError("Expected IMAGE tensor in shape [B, H, W, C].")

        b, h, w, c = image.shape
        _ = (b, c)  # unused, for readability

        x1 = max(0, min(int(x1), w - 1))
        x2 = max(1, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(1, min(int(y2), h))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid wall bbox: ensure x2>x1 and y2>y1.")

        wall_crop = image[:, y1:y2, x1:x2, :]
        wall_pixel_width = int(x2 - x1)
        return (wall_crop, wall_pixel_width)


class PixelScaleFromRoomSizeNode:
    """
    Step 3: use known room size to build meter-per-pixel scale.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wall_pixel_width": ("INT", {"default": 100, "min": 1, "max": 1000000, "step": 1}),
                "wall_real_width_m": ("FLOAT", {"default": 3.0, "min": 0.0001, "max": 10000.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("meter_per_pixel", "pixel_per_meter")
    FUNCTION = "build_scale"
    CATEGORY = "Panorama/Measurement"

    def build_scale(self, wall_pixel_width, wall_real_width_m):
        wall_pixel_width = float(wall_pixel_width)
        wall_real_width_m = float(wall_real_width_m)

        if wall_pixel_width <= 0:
            raise ValueError("wall_pixel_width must be > 0")
        if wall_real_width_m <= 0:
            raise ValueError("wall_real_width_m must be > 0")

        meter_per_pixel = wall_real_width_m / wall_pixel_width
        pixel_per_meter = wall_pixel_width / wall_real_width_m
        return (float(meter_per_pixel), float(pixel_per_meter))


class TargetPixelWidthNode:
    """
    Step 4: measure target object's pixel width by bbox.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_x1": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "target_x2": ("INT", {"default": 100, "min": 1, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("target_pixel_width",)
    FUNCTION = "measure_target_width"
    CATEGORY = "Panorama/Measurement"

    def measure_target_width(self, image, target_x1, target_x2):
        if image.ndim != 4:
            raise ValueError("Expected IMAGE tensor in shape [B, H, W, C].")

        _, _, w, _ = image.shape
        x1 = max(0, min(int(target_x1), w - 1))
        x2 = max(1, min(int(target_x2), w))

        if x2 <= x1:
            raise ValueError("Invalid target bbox: ensure target_x2 > target_x1")

        return (int(x2 - x1),)


class PixelToRealSizeNode:
    """
    Step 5: convert target pixel width to real-world size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_pixel_width": ("INT", {"default": 100, "min": 1, "max": 1000000, "step": 1}),
                "meter_per_pixel": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10000.0, "step": 0.0001}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("target_width_m", "target_width_cm")
    FUNCTION = "convert"
    CATEGORY = "Panorama/Measurement"

    def convert(self, target_pixel_width, meter_per_pixel):
        target_pixel_width = float(target_pixel_width)
        meter_per_pixel = float(meter_per_pixel)

        if target_pixel_width <= 0:
            raise ValueError("target_pixel_width must be > 0")
        if meter_per_pixel <= 0:
            raise ValueError("meter_per_pixel must be > 0")

        target_width_m = target_pixel_width * meter_per_pixel
        target_width_cm = target_width_m * 100.0
        return (float(target_width_m), float(target_width_cm))


NODE_CLASS_MAPPINGS = {
    "PanoramaWallCrop": PanoramaWallCropNode,
    "PixelScaleFromRoomSize": PixelScaleFromRoomSizeNode,
    "TargetPixelWidth": TargetPixelWidthNode,
    "PixelToRealSize": PixelToRealSizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoramaWallCrop": "Panorama Wall Crop (BBox)",
    "PixelScaleFromRoomSize": "Pixel Scale From Room Size",
    "TargetPixelWidth": "Target Pixel Width (BBox)",
    "PixelToRealSize": "Pixel To Real Size",
}
