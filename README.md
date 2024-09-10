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
