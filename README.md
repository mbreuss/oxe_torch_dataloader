# Uha
![Uha Logo](img/uha_logo.png)
> *Uha*: Minimal adaption of the dataloader proposed in Octo for loading over 25 embodiments.

Package repository for Uha: Load and adjust Open X-Embodiement Datasets like Octo.

---

## Installation

```bash
git clone https://github.com/mbreuss/openrt_torch_dataloader
cd openrt_torch_dataloader
conda create -n uha python=3.10
conda activate uha
pip install -e .
```

## Usage

Provides a minimal setup to load the `pytorch_oxe_dataloader` from Octo without additional content.
Simply install this repo via pip and use `make_pytorch_oxe_iterable_dataset(dataset, batch_size)`.
