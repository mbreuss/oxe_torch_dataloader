# Uha
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<img src="/img/uha_logo.png" alt="Uha Logo" width="512" height="auto">

> *Uha*: Minimal adaption of the dataloader proposed in Octo for loading over 25 embodiments.

Package repository for Uha: Load and adjust Open X-Embodiement Datasets like Octo.
**Work in Progress, currently only Public because we need it for another project.**

---

## Installation

```bash
git clone https://github.com/mbreuss/openrt_torch_dataloader
cd openrt_torch_dataloader
conda create -n uha python=3.10
conda activate uha
pip install -e .
```

---

## Usage

Provides a minimal setup to load the `pytorch_oxe_dataloader` from Octo without additional content.
Simply install this repo via pip and use `make_pytorch_oxe_iterable_dataset(dataset, batch_size)`.

---

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
