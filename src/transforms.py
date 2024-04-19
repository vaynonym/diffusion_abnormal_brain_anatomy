from monai import transforms

def get_crop_around_mask_center(run_config):
    def crop_around_mask_center(d):
        is_not_background = (d["mask"] > 0.0).nonzero()
        max_indices, _ = is_not_background[:, 1:].max(dim=0)
        min_indices, _= is_not_background[:, 1:].min(dim=0)

        center = min_indices + (max_indices - min_indices) // 2

        crop = transforms.SpatialCrop(roi_center=center, roi_size=run_config["input_image_crop_roi"])
        pad = transforms.SpatialPad(spatial_size=run_config["input_image_crop_roi"], value=0)

        return {
            **d,
            "image": pad(crop(d["image"])),
            "mask": pad(crop(d["mask"])),
        }
    return crop_around_mask_center