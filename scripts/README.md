# Scripts

Useful scripts for various functionality (documented below). Requires installing the source package locally!

## test_loader
Simple script to test the loader and log its output. In the code, there are 2 versions provided: ```make_pytorch_oxe_iterable_dataset(...)``` with transform_dict and without. The one without ```transform_dict``` outputs the default octo dataloading, the other takes a dict as input (instantiated hydra OmegaConf or hand made dict) and adjusts the loading. Examples for the configs can be found [in the conf folder](../uha/data/conf).