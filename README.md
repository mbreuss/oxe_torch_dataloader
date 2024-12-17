# Uha
<img src="/img/uha_logo.png" alt="Uha Logo" width="850" height="auto">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *Uha*: Minimal adaption of the dataloader proposed in Octo for loading over 25 embodiments.

Package repository for Uha: Load and adjust Open X-Embodiement Datasets like Octo.

---

## Installation

```bash
git clone [https://github.com/mbreuss/openrt_torch_dataloader](https://github.com/mbreuss/oxe_torch_dataloader.git)
cd oxe_torch_dataloader
conda create -n uha python=3.9
conda activate uha
pip install -e .
```


# Technical Guide

## Architecture Overview

The Uha dataloader is structured into several key components that work together to load and process robotic datasets from the Open X-Embodiment collection. Here's a detailed breakdown of how it works:

### Core Components

#### 1. Data Source Management (`/uha/data/oxe/`)
The `oxe` directory contains the core configuration and dataset management code:

- `oxe_dataset_configs.py`: Defines all supported datasets and their configurations (camera views, action/proprio encodings)
- `oxe_dataset_mixes.py`: Specifies predefined combinations of datasets with sampling weights
- `oxe_standardization_transforms.py`: Contains transforms to convert each dataset into a standardized format

#### 2. Data Processing Pipeline (`/uha/data/utils/`)
The utils directory handles the data processing pipeline:

- `data_utils.py`: Core utilities for data manipulation and normalization
- `dataset_index.py`: Maintains mappings between dataset names and indices
- `goal_relabeling.py`: Handles relabeling of goal states for trajectory data
- `rlds_utils.py`: Utilities for working with the RLDS (Robot Learning Data Structure) format
- `spec.py`: Defines specifications for modules and transformations
- `task_augmentation.py`: Implements data augmentation strategies

#### 3. Transform Layers (`/uha/data/`)
Two main types of transforms are implemented:

- `obs_transforms.py`: Frame-level transformations (image processing, resizing)
- `traj_transforms.py`: Trajectory-level transformations (chunking, padding)

#### 4. PyTorch Integration (`/uha/`)
The core PyTorch integration happens in:

- `pytorch_oxe_dataloader.py`: PyTorch IterableDataset implementations
- `uha_datamodule.py`: PyTorch Lightning DataModule implementations

## How the Dataloader Works

### 1. Dataset Configuration

The process begins with dataset configuration. Let's look at a typical configuration:

```python
# Example dataset config from oxe_dataset_configs.py
"bridge": {
    "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "proprio_encoding": ProprioEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
    "language_key": "groundtruth*"
}
```

### 2. Data Loading Pipeline

The loading process follows these steps:

1. **Dataset Initialization**: `make_dataset_from_rlds()` loads the raw dataset
2. **Standardization**: The dataset is converted to a standard format using transforms
3. **Trajectory Processing**: Applies chunking, padding, and goal relabeling
4. **Frame Processing**: Handles image transformations and augmentation
5. **PyTorch Conversion**: Data is converted to PyTorch tensors and batched

### 3. Using the Dataloader

Here's a basic example of using the dataloader:

```python
from uha.uha_datamodule import UhaDataModule
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize datamodule
    datamodule = UhaDataModule(
        datasets=cfg.datasets,
        batch_size=32,
        num_workers=4,
        transforms=cfg.transforms,
        language_encoders=cfg.language_encoders
    )
    
    # Create dataloaders
    train_loader = datamodule.create_train_dataloader()
    val_loader = datamodule.create_val_dataloader()
```

## Key Features

### 1. Multi-Dataset Support
The dataloader can handle multiple datasets simultaneously with different:
- Camera configurations
- Action spaces
- Proprioceptive states
- Language annotations

### 2. Standardization
All datasets are converted to a standard format with:
- Normalized actions and proprioceptive states
- Consistent image formats
- Unified trajectory structures

### 3. Efficient Processing
The dataloader implements several efficiency features:
- Lazy loading of images
- Efficient memory management
- Parallel data processing
- Configurable prefetching

## Adding New Datasets

To add a new dataset:

1. **Configure Dataset**: Add dataset configuration to `oxe_dataset_configs.py`:
```python
"new_dataset": {
    "image_obs_keys": {...},
    "depth_obs_keys": {...},
    "proprio_encoding": ProprioEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
}
```

2. **Create Transform**: Add standardization transform to `oxe_standardization_transforms.py`:
```python
def new_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # Transform implementation
    return transformed_trajectory
```

3. **Add to Mix**: Include in dataset mix in `oxe_dataset_mixes.py`:
```python
NEW_DATASET_MIX = [
    ("new_dataset", 1.0),
    ("existing_dataset", 0.5),
]
```

## Common Configurations

### Data Transforms
Configure transforms in your config file:
```yaml
transforms:
  move_axis: True
  bytes_to_string: True
  adjust_type: torch.float32
  key_remapping:
    observation:
      image_primary: ["rgb_obs", "rgb_static"]
      proprio: "robot_obs"
```

### Language Encoders
Available language encoders:
- CLIP (`clip.yaml`)
- Florence (`florence_large.yaml`)
- No encoding (`no_encoder.yaml`)

## Debugging Tips

1. Use the provided test scripts:
   - `scripts/test_loader.py`
   - `scripts/print_stats.py`

2. Enable verbose logging for debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

3. Check dataset statistics:
```python
datamodule = UhaDataModule(...)
stats = datamodule.get_dataset_statistics()
print(stats["train_dataset"])
```

This guide should help you understand and work with the Uha dataloader effectively. For more detailed information about specific components, refer to the docstrings in the respective files.


## Usage

Provides a minimal setup to load the [pytorch_oxe_dataloader from Octo](https://github.com/octo-models/octo/blob/main/examples/06_pytorch_oxe_dataloader.py) without additional content.
To utilize this repository, include it as a submodule in git and load [the Uha Datamodule](uha/uha_datamodule.py).

## Adding New Datasets
To add a new dataset, first convert it to RLDS by following [this example repo](https://github.com/Toradus/rlds_dataset_builder). Once the dataset is converted, it will be stored in "~/tensorflow_datasets/[DATASET_NAME]". To add it to Uha, add a new line in [the OXE_DATASET_CONFIGS](uha/data/oxe/oxe_dataset_configs.py) with "data_dir" pointing towards the "~/tensorflow_datasets" location (different examples are in the file). Afterwards, add a new SymbolicTensor standardization transform to [the OXE_STANDARDIZATION_TRANSFORMS](uha/data/oxe/oxe_standardization_transforms.py) and add your dataset to [a new mixture](uha/data/oxe/oxe_dataset_mixes.py).

To verify that the dataset is converted correctly, you can execute [the tfds quickcheck](scripts/quick_check_tfds.ipynb), which loads the raw data via tfds, or [the dataloader quickcheck](scripts/quick_check_dataloader.ipynb), which loads the standardized data via the dataloader. For the dataloader quickcheck, [the hydra config](uha/data/conf/uha_default_load_config.yaml) "DATA_NAME" property has to be adjusted, according to the new mixture name.

## Acknowledgements

Most of the code is based on the [Octo Codebase](https://github.com/octo-models/octo) and has just been isoloated and adapted for Torch applications.
Thus, if you find this usefull, please cite their work:

```
@misc{octo_2023,
    title={Octo: An Open-Source Generalist Robot Policy},
    author = {{Octo Model Team} and Dibya Ghosh and Homer Walke and Karl Pertsch and Kevin Black and Oier Mees and Sudeep Dasari and Joey Hejna and Charles Xu and Jianlan Luo and Tobias Kreiman and {You Liang} Tan and Dorsa Sadigh and Chelsea Finn and Sergey Levine},
    howpublished  = {\url{https://octo-models.github.io}},
    year = {2023},
}
```

--- 
