import torch
from torch import Tensor
from src.torch_setup import device
from generative.networks.nets.autoencoderkl import AutoencoderKL 
import torch.nn as nn
from typing import Tuple

synthseg_classes = [ 0,  2,  3,  4,  5,  7,  8, 10, 11,
                     12, 13, 14, 15, 16, 17, 18, 24, 26,
                     28, 41, 42, 43, 44, 46, 47, 49, 50,
                     51, 52, 53, 54, 58, 60]

synthseg_index_mask_value_map = {idx:c for idx, c in enumerate(synthseg_classes)}

ventricles = [
    "right lateral ventricle",
    "right inferior lateral ventricle",
    "left lateral ventricle",
    "left inferior lateral ventricle",
    "3rd ventricle",
    "4th ventricle",
]

def reduce_mask_to_ventricles(mask: Tensor, device=device):
    ventricle_indices = torch.tensor([index for index, name in synthseg_class_to_string_map.items() 
                                      if name in ventricles]).to(device)

    return torch.isin(mask, ventricle_indices)

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

def encode_one_hot(mask: torch.Tensor, device=device):
    shp = list(mask.shape)
    shp[1] = len(synthseg_classes)
    one_hot = torch.zeros(shp).to(device)
    for channel in range(len(synthseg_classes)):
        one_hot[:, channel, ...] = mask[:, 0, ...] == synthseg_index_mask_value_map[channel]
    
    return one_hot

def decode_one_hot(mask: torch.Tensor, device=device):
    # [B, C, SPATIALS]
    assert len(list(mask.shape)) == 5
    unhotted = mask.argmax(1, keepdim=True)

    base = torch.arange(0, len(synthseg_classes)).to(device)
    classes= torch.tensor(synthseg_classes).to(device)

    index = torch.bucketize(unhotted.ravel(), base)

    return (classes[index].reshape(unhotted.shape)).to(torch.int32)


