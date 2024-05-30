# Guided Synthesis of Labeled Brain MRI Data Using Latent Diffussion Models for Segmentation of Enlarged Ventricles

This is the accompanying code repository for the Master Thesis of Tim Ruschke. The thesis revolves around the use of two LDMs, a mask generator and an image generator, which are used to create fully synthetic data. We manage to beat state-of-the-art model SynthSeg on a downstream task using nnUNets trained exclusively with synthetic data. The task consists of the segmentation of ventricles for patients suffering from NPH. 

# Environment

See `environment.yml` for a snapshot of all libraries used.

Many parts of the repository use WandB to log data. As such, we expect a file titled `local_config.yml` with the two keys `project` and `entity` to be set. These identify the WandB project and account used, respectively.

Additionally, the following environment variables to folder paths are expected to be set. Choose whichever folder paths you prefer.

`DATA_DIRECTORY` defines the location of the the datasets. Depending on the script executed, certain subpaths may be expected to exist, such as the LDM100k dataset.

`FLAIR_DATASET_DIRECTORY` is an extra variable defining specifically the location of the flair dataset. This is for cases where the data may not be locally available.

`OUTPUT_DIRECTORY` defines the directory for program output like figures.

`MODEL_DIRECTORY` is where models will be saved during training, or where checkpoints are expected to reside for continuing.

`PRETRAINED_MODEL_DIRECTORY` is the location for the pretrained weights for the model on which FID is based.

`EXPERIMENT_CONFIG` selects the particular config of the experiment

`WANDB_LOG_IMAGES` one or undefined. Can be used to deactivate some WandB logging.

# Structure

`scripts` define top-level scripts for training, evaluation or inference. `src` contains supporting python files defining much of the meat of the project. `configs` define the model as well as some hyperparameters for its training.