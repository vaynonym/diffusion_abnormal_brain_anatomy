#!/usr/bin/env python3

# Set up imports
import os
import torch
import torch.nn.functional as F
from monai import transforms
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from src.evaluation import create_fake_volume_dataloader
from src.util import load_wand_credentials, Stopwatch, read_config, read_config_from_wandb_run
from src.model_util import load_model_from_run_with_matching_config, load_model_from_run, check_dimensions
from src.logging_util import LOGGER
from src.directory_management import DATA_DIRECTORY, OUTPUT_DIRECTORY
from src.datasets import get_dataloader, get_default_transforms, load_dataset_from_config


from src.diffusion import get_scale_factor
from src.custom_autoencoders import EmbeddingWrapper
from src.synthseg_masks import synthseg_classes, central_areas_close_to_ventricles_indices, ventricle_indices, white_matter_indices, cortex_indices, CSF_indices, background_indices, decode_one_hot
from src.evaluation import MaskDiffusionModelEvaluator



import torch.multiprocessing


from src.torch_setup import device

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER.info(f"Device count: {torch.cuda.device_count()}", )
LOGGER.info(f"Device: {device}")

WANDB_LOG_IMAGES = os.environ.get("WANDB_LOG_IMAGES")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME")
WANDB_AUTOENCODER_RUNID = os.environ.get("WANDB_AUTOENCODER_RUNID")
WANDB_DIFFUSIONMODEL_RUNID = os.environ.get("WANDB_DIFFUSIONMODEL_RUNID")


project, entity = load_wand_credentials()

#experiment_config = read_config(os.environ.get("EXPERIMENT_CONFIG"))
experiment_config = read_config_from_wandb_run(entity, project, WANDB_RUN_NAME)

run_config = experiment_config["run"]
run_config["oversample_large_ventricles"] = False
LOGGER.info(f"Oversample large ventricles: {run_config['oversample_large_ventricles']}")

auto_encoder_config = experiment_config["auto_encoder"]
patch_discrim_config = experiment_config["patch_discrim"]
auto_encoder_training_config = experiment_config["autoencoder_training"]
diffusion_model_unet_config = experiment_config["diffusion_model_unet"]
diffusion_model_training_config = experiment_config["diffusion_model_unet_training"]

check_dimensions(run_config, auto_encoder_config, diffusion_model_unet_config)

CLASSIFIER_FREE_GUIDANCE = diffusion_model_training_config["classifier_free_guidance"] if "classifier_free_guidance" in diffusion_model_training_config else None
LOGGER.info(f"Using classifier free guidance: {CLASSIFIER_FREE_GUIDANCE}")


# for reproducibility purposes set a seed
set_determinism(42)

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
image_shape = (run_config["batch_size"], 1, run_config["input_image_crop_roi"][0], run_config["input_image_crop_roi"][1], run_config["input_image_crop_roi"][2])
LOGGER.info(f"Image shape: {image_shape}")
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))

LOGGER.info("Loading models...")

def create_embedding_autoencoder(*args, **kwargs):
    base_autoencoder = AutoencoderKL(*args, **kwargs)
    autoencoder = EmbeddingWrapper(base_autoencoder=base_autoencoder, vocabulary_size=max(synthseg_classes) + 1, embedding_dim=64)
    return autoencoder

autoencoder = load_model_from_run(run_id=WANDB_AUTOENCODER_RUNID, project=project, entity=entity,
                                  #model_class=AutoencoderKL,
                                  model_class=EmbeddingWrapper,  
                                  #create_model_from_config=None,
                                  create_model_from_config=create_embedding_autoencoder
                                  )


diffusion_model = load_model_from_run(run_id=WANDB_DIFFUSIONMODEL_RUNID, project=project, entity=entity,
                                      model_class=DiffusionModelUNet, 
                                      create_model_from_config=None
                                     )

LOGGER.info("Loading dataset...")

_, valid_transforms = get_default_transforms(run_config, CLASSIFIER_FREE_GUIDANCE)
validation_ds = load_dataset_from_config(run_config, "validation", valid_transforms)

from src.dataset_analysis import log_bucket_counts

log_bucket_counts(validation_ds)

valid_loader = get_dataloader(validation_ds, run_config)

LOGGER.info(f"Valid length: {len(validation_ds)} in {len(valid_loader)} batches")
LOGGER.info(f'Mask shape {validation_ds[0]["mask"].shape}')


stopwatch = Stopwatch("Calculating volumes over validationset took: ").start()
bucket_size = 0.1
buckets = torch.tensor(np.arange(0, 1.0, bucket_size)).to(device)
# ensure that every normal value is within 0 to 1 buckets, essentially making [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0], (1.0, 1.1], ...
buckets[0] = -0.001 

classes = [
    ("ventricles", ventricle_indices, {i:[] for i in range(len(buckets)+1)}),
    ("white_matter", white_matter_indices, {i:[] for i in range(len(buckets)+1)}),
    ("CSF", CSF_indices, {i:[] for i in range(len(buckets)+1)}),
    ("cortex", cortex_indices , {i:[] for i in range(len(buckets)+1)}),
    ("background", background_indices, {i:[] for i in range(len(buckets)+1)}),
    ("other central areas", central_areas_close_to_ventricles_indices, {i:[] for i in range(len(buckets)+1)}),
]

figures = [plt.figure(i + 1) for i, _ in enumerate(classes)]

def append_voxel_count(mask, buckets_of_batch, relevant_indices, result_voxel_counts):
    voxel_count = torch.isin(mask, relevant_indices).sum(dim=(1,2,3,4))
    for batch_i in range(mask.shape[0]):
        result_voxel_counts[buckets_of_batch[batch_i].item()].append(voxel_count[batch_i].float())
    return result_voxel_counts

for i, batch in enumerate(valid_loader):
    mask = batch["mask"].int().to(device)
    conditioning_value = batch["volume"].to(device)
    buckets_of_batch = torch.bucketize(input=conditioning_value[:, :, 0], boundaries=buckets)

    for batch_i in range(mask.shape[0]):
        for (name, indices, voxel_counts) in classes:
            append_voxel_count(mask, buckets_of_batch, indices, voxel_counts)



for (name, _ , voxel_counts) in classes:
    maxes = [torch.stack(v).max().item() if len(v) > 0 else 0 for k, v in voxel_counts.items()]
    LOGGER.info(f"Max {name}: {np.array(maxes)}")
    LOGGER.info(f"Real (Bin, #Samples): {[(i, len(x)) for i, x in enumerate(voxel_counts.values())]}")


results = [(name,
            {k:torch.stack(v).mean().item() if len(v) > 0 else 0 for k, v in voxel_counts.items()}, 
            {k:(torch.stack(v).mean().item() - torch.stack(v).min().item() if len(v) > 0 else 0,
                torch.stack(v).max().item() - torch.stack(v).mean().item() if len(v) > 0 else 0,
                ) for k, v in voxel_counts.items()}, 
            ) for (name, _, voxel_counts) in classes]

stopwatch.stop().display()


stopwatch = Stopwatch("drawing plots took: ").start()



for fig, (name, mean_voxel_counts, min_max_voxel_counts) in zip(figures, results):
    min_max_error = np.array(list(min_max_voxel_counts.values())).T
    plt.figure(fig)
    plt.errorbar(x=np.arange(0, 1.0, bucket_size) + bucket_size/2,
            #width=bucket_size,
            y=list(mean_voxel_counts.values())[1:],
            #yerr=min_max_error[:, 1:],
            color="blue",
            label="Ground Truth",
            marker="o", capsize=5, capthick=1, ecolor="blue", lw=1
            )

    LOGGER.info(f"{name}: {np.array(list(mean_voxel_counts.values()))}")

stopwatch.stop().display()

LOGGER.info("DONE!")

scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
scheduler.set_timesteps(50)

LOGGER.info(f"Using scheduler {scheduler.__class__.__name__} with timesteps {scheduler.num_inference_steps}")

#scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data= encode_one_hot(sample_data["mask"].to(device)))

scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data=next(iter(valid_loader))["mask"].to(device))

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

autoencoder.eval()

# include abnormal values
buckets = torch.tensor(np.arange(0, 1.6, bucket_size)).to(device)
# ensure that every normal value is within 0 to 1 buckets, essentially making [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0], (1.0, 1.1], ...
buckets[0] = -0.001 

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])

LOGGER.info(f"Encoding shape: {encoding_shape}")
LOGGER.info("Generating synthetic examples...")

fake_volume_dataloader = create_fake_volume_dataloader(min_val=1.05, max_val=1.6, bucket_size=0.1, number_samples_per_bucket=50,
                                                       classifier_free_guidance=CLASSIFIER_FREE_GUIDANCE,
                                                       batch_size=run_config["batch_size"]
                                                       )

LOGGER.info(f"Fake volume dataloader length: {len(fake_volume_dataloader)}")

guidance_values = [1.0, 2.0, 3.0, 4.0, 5.0]


legend_labels = ["Ground Truth"] + [f"Synthetic G={G:.1f}" for G in guidance_values]

plt.legend(legend_labels)

guidance_all_results = []

for G in guidance_values:
    LOGGER.info(f"Starting calculation for G={G}")

    classes = [
        ("ventricles", ventricle_indices, {i:[] for i in range(len(buckets) + 1)}),
        ("white_matter", white_matter_indices, {i:[] for i in range(len(buckets) + 1)}),
        ("CSF", CSF_indices, {i:[] for i in range(len(buckets) +1)}),
        ("cortex", cortex_indices , {i:[] for i in range(len(buckets)+1)}),
        ("background", background_indices, {i:[] for i in range(len(buckets)+1)}),
        ("other central areas", central_areas_close_to_ventricles_indices, {i:[] for i in range(len(buckets)+1)}),
    ]

    evaluator = MaskDiffusionModelEvaluator(
                 diffusion_model=diffusion_model,
                 autoencoder=autoencoder,
                 latent_shape=encoding_shape,
                 inferer=inferer,
                 val_loader=valid_loader,
                 train_loader=None,
                 wandb_prefix="diffusion/evaluation",
                 evaluation_scheduler=scheduler,
                 guidance=G
                 )

    for batch in tqdm(valid_loader, desc="Generating for val set"):
        conditioning_value = batch["volume"].to(device)
        mask = decode_one_hot(evaluator.get_synthetic_output(batch, True)).int()

        buckets_of_batch = torch.bucketize(input=conditioning_value[:, :, 0], boundaries=buckets)

        for batch_i in range(mask.shape[0]):
            for (name, indices, voxel_counts) in classes:
                append_voxel_count(mask, buckets_of_batch, indices, voxel_counts)

    for batch in tqdm(fake_volume_dataloader, desc="Generating for abnormal values"):
        conditioning_value = batch["volume"].to(device)
        mask = decode_one_hot(evaluator.get_synthetic_output(batch, True)).int()


        buckets_of_batch = torch.bucketize(input=conditioning_value[:, :, 0], boundaries=buckets)

        for batch_i in range(mask.shape[0]):
            for (name, indices, voxel_counts) in classes:
                append_voxel_count(mask, buckets_of_batch, indices, voxel_counts)


    LOGGER.info(f"Synthetic: (Bin, #Samples): {[(i, len(x)) for i, x in enumerate(voxel_counts.values())]}")

    for (name, _ , voxel_counts) in classes:
        maxes = [torch.stack(v).max().item() if len(v) > 0 else 0 for k, v in voxel_counts.items()]
        LOGGER.info(f"Max {name}: {np.array(maxes)}")
        LOGGER.info(f"Real (Bin, #Samples): {[(i, len(x)) for i, x in enumerate(voxel_counts.values())]}")

    results = [(name,
                {k:torch.stack(v).mean().item() if len(v) > 0 else 0 for k, v in voxel_counts.items()}, 
                {k:(torch.stack(v).mean().item() - torch.stack(v).min().item() if len(v) > 0 else 0,
                    torch.stack(v).max().item() - torch.stack(v).mean().item() if len(v) > 0 else 0,
                    ) for k, v in voxel_counts.items()}, 
                ) for (name, _, voxel_counts) in classes]
    
    guidance_all_results.append(results)

colors = ["lightcoral",
          "indianred",
          "brown",
          "firebrick",
          "maroon",
          "rosybrown",
          ]

for G, results, color in zip(guidance_values, guidance_all_results, colors):
    LOGGER.info(f"======== RESULTS FOR G={G} ========")
    for fig, (name, mean_voxel_counts, min_max_voxel_counts) in zip(figures, results):
        plt.figure(fig)

        min_max_error = np.array(list(min_max_voxel_counts.values())).T
        plt.errorbar(x=np.arange(0, 1.6, bucket_size) + bucket_size/2,
                #width=bucket_size,
                y=list(mean_voxel_counts.values())[1:],
                #yerr=min_max_error[:, 1:],
                color=color,
                label=f"Synthetic G={G:.1f}",
                marker="o", capsize=5, capthick=1, ecolor="red", lw=1, alpha=0.8
                )

        LOGGER.info(f"{name}: \n{repr(np.array(list(mean_voxel_counts.values())))}")

for fig, (name, _, _ ) in zip(figures, classes):
    plt.figure(fig)
    plt.xlabel("Normalized Volume Ratio", fontsize=16)
    plt.ylabel("Voxel Count", fontsize=16)
    plt.figlegend()
    plt.savefig(f"{OUTPUT_DIRECTORY}/{name}_volume.png")
