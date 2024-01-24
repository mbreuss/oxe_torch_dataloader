# minimum working example to load the OXE dataset
from uha import make_pytorch_oxe_iterable_dataset, get_octo_dataset_tensorflow, get_octo_dataset_tensorflow_test
import tqdm
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../uha/data/conf", config_name="uha_default_load_config")
def main(cfg: DictConfig):
    dataset = get_octo_dataset_tensorflow_test(cfg, True)
    dataloader = make_pytorch_oxe_iterable_dataset(dataset)
    for i, sample in tqdm.tqdm(enumerate(dataloader)):
        print(sample)
        if i == 5000:
            break


if __name__ == "__main__":
    main()
