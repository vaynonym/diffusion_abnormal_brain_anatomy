import torch
from torch import Tensor
from src.torch_setup import device
from typing import Callable


synthseg_classes = [ 0,  2,  3,  4,  5,  7,  8, 10, 11,
                     12, 13, 14, 15, 16, 17, 18, 24, 26,
                     28, 41, 42, 43, 44, 46, 47, 49, 50,
                     51, 52, 53, 54, 58, 60]

synthseg_index_mask_value_map = {idx:c for idx, c in enumerate(synthseg_classes)}

def reduce_mask_to_ventricles(mask: Tensor, device=device):
    ventricle_indices = torch.tensor([index for index, name in synthseg_class_to_string_map.items() 
                                      if name in ventricles]).to(device)

    return torch.isin(mask, ventricle_indices)

def get_max_ventricle_crop(dataloader):
    (i, first) = next(enumerate(dataloader))
    mask : Tensor = first["mask"]
    assert len(list(mask.shape)) == 5
    shape = mask.shape
    max_x, max_y, max_z  = tuple(shape[2:] // 2)
    min_x, min_y, min_z  = tuple(shape[2:] // 2)
    del mask

    for _, batch in enumerate(dataloader):
        ventricle_mask = reduce_mask_to_ventricles(batch["mask"])
        ventricle_indices = ventricle_mask.nonzero()

        local_max_x = torch.max(ventricle_indices[:, 2]).item()
        local_min_x = torch.min(ventricle_indices[:, 2]).item()
        local_max_y = torch.max(ventricle_indices[:, 3]).item()
        local_min_y = torch.min(ventricle_indices[:, 3]).item()
        local_max_z = torch.max(ventricle_indices[:, 4]).item()
        local_min_z = torch.min(ventricle_indices[:, 4]).item()

        max_x = max(max_x, local_max_x)
        min_x = min(min_x, local_min_x)
        max_y = max(max_y, local_max_y)
        min_y = min(min_y, local_min_y)
        max_z = max(max_z, local_max_z)
        min_z = min(min_z, local_min_z)

    margin = 5
    max_x = min(max_x + margin, shape[2])
    min_x = max(min_x - margin, 0)
    max_y = min(max_y + margin, shape[3])
    min_y = max(min_y - margin, 0)
    max_z = min(max_z + margin, shape[4])
    min_z = max(min_z - margin, 0)
    
    return (max_x, min_x), (max_y, min_y), (max_z, min_z)

def get_crop_to_max_ventricle_shape(dataloader) -> Callable[[dict], dict]:
    (max_x, min_x), (max_y, min_y), (max_z, min_z) = get_max_ventricle_crop(dataloader)

    return lambda x: x[:, :, min_x:max_x, min_y:max_y, min_z:max_z]

synthseg_class_to_string_map = {
0:         "background",
2:         "left cerebral white matter",
3:         "left cerebral cortex",
4:         "left lateral ventricle",
5:         "left inferior lateral ventricle",
7:         "left cerebellum white matter",
8:         "left cerebellum cortex",
10:        "left thalamus",
11:        "left caudate",
12:        "left putamen",
13:        "left pallidum",
14:        "3rd ventricle",
15:        "4th ventricle",
16:        "brain-stem",
17:        "left hippocampus",
18:        "left amygdala",
26:        "left accumbens area",
24:        "CSF",
28:        "left ventral DC",
41:        "right cerebral white matter",
42:        "right cerebral cortex",
43:        "right lateral ventricle",
44:        "right inferior lateral ventricle",
46:        "right cerebellum white matter",
47:        "right cerebellum cortex",
49:        "right thalamus",
50:        "right caudate",
51:        "right putamen",
52:        "right pallidum",
53:        "right hippocampus",
54:        "right amygdala",
58:        "right accumbens area",
60:        "right ventral DC",
}

central_areas_close_to_ventricles = [
    "left thalamus",
    "left caudate",
    "left putamen",
    "left pallidum",
    "left hippocampus",
    "left amygdala",
    "left accumbens area",
    "left ventral DC",
    "right thalamus",
    "right caudate",
    "right putamen",
    "right pallidum",
    "right hippocampus",
    "right amygdala",
    "right accumbens area",
    "right ventral DC",
]
central_areas_close_to_ventricles_indices  = torch.tensor([index for (index, name) in synthseg_class_to_string_map.items() 
                                                                 if name in central_areas_close_to_ventricles]).to(device)
assert len(central_areas_close_to_ventricles_indices) == 16

ventricles = [
    "right lateral ventricle",
    "right inferior lateral ventricle",
    "left lateral ventricle",
    "left inferior lateral ventricle",
    "3rd ventricle",
    "4th ventricle",
]
ventricle_indices  = torch.tensor([index for index, name in synthseg_class_to_string_map.items() 
                                      if name in ventricles]).to(device)
assert len(ventricle_indices) == 6


white_matter = [
    "left cerebral white matter",
    "left cerebellum white matter",
    "right cerebral white matter",
    "right cerebellum white matter",
]
white_matter_indices = torch.tensor([index for index, name in synthseg_class_to_string_map.items() 
                                      if name in white_matter]).to(device)
assert len(white_matter_indices) == 4


background = ["background"]
background_indices = torch.tensor([index for index, name in synthseg_class_to_string_map.items() 
                                      if name in background]).to(device)
assert len(background_indices) == 1

cortex = [
          "right cerebellum cortex",
          "right cerebral cortex",
          "left cerebellum cortex",
          "left cerebral cortex",
          ]
cortex_indices = torch.tensor([index for index, name in synthseg_class_to_string_map.items() 
                                      if name in cortex]).to(device)
assert len(cortex_indices) == 4

CSF = ["CSF"]
CSF_indices = torch.tensor([index for index, name in synthseg_class_to_string_map.items() 
                                      if name in CSF]).to(device)
assert len(CSF_indices) == 1

def encode_one_hot(mask: torch.Tensor, device=device):
    shp = list(mask.shape)
    shp[1] = len(synthseg_classes)
    one_hot = torch.zeros(shp).to(device)
    for channel in range(len(synthseg_classes)):
        one_hot[:, channel, ...] = mask[:, 0, ...] == synthseg_index_mask_value_map[channel]
    
    return one_hot

# decodes to actual synthseg values
def decode_one_hot(mask: torch.Tensor, device=device):
    # [B, C, SPATIALS]
    assert len(list(mask.shape)) == 5
    unhotted = mask.argmax(1, keepdim=True)

    base = torch.arange(0, len(synthseg_classes)).to(device)
    classes= torch.tensor(synthseg_classes).to(device)

    index = torch.bucketize(unhotted.ravel(), base)

    return (classes[index].reshape(unhotted.shape)).to(torch.int32)


