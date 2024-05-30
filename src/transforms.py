from monai import transforms

def get_crop_around_mask_center(crop_size):
    def crop_around_mask_center(d):
        is_not_background = (d["mask"] > 0.0).nonzero()
        if is_not_background.shape[0] == 0:
            raise Exception(f"Mask is Empty: {d['case_identifier']}")
        
        max_indices, _ = is_not_background[:, 1:].max(dim=0)
        min_indices, _= is_not_background[:, 1:].min(dim=0)

        center = min_indices + (max_indices - min_indices) // 2

        crop = transforms.SpatialCrop(roi_center=center, roi_size=crop_size)
        pad = transforms.SpatialPad(spatial_size=crop_size, value=0)


        result = {
            **d,
            "image": pad(crop(d["image"])),
            "mask": pad(crop(d["mask"])),
        }

        if "synthseg_mask" in d:
            result = {**result, "synthseg_mask": pad(crop(d["synthseg_mask"]))}
        
        return result


    return crop_around_mask_center