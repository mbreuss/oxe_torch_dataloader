# minimum working example to load the OXE dataset
from uha import make_pytorch_oxe_iterable_dataset, get_octo_dataset_tensorflow, get_octo_dataset_tensorflow_old
import tqdm
import hydra
from omegaconf import DictConfig


def main(cfg: DictConfig):
    dataset = get_octo_dataset_tensorflow_old(True)
    dataloader = make_pytorch_oxe_iterable_dataset(dataset)
    for i, sample in tqdm.tqdm(enumerate(dataloader)):
        print(sample)
        if i == 5000:
            break


if __name__ == "__main__":
    main()
