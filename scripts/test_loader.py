# minimum working example to load the OXE dataset
from uha import make_pytorch_oxe_iterable_dataset, get_octo_dataset_tensorflow
import tqdm
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../uha/data/conf", config_name="uha_default_load_config")
def main(cfg: DictConfig):
    # load Training Dataset from TensorflowDatasets
    dataset = get_octo_dataset_tensorflow(cfg, train=True)
    # create Pytorch Train Dataset
    dataloader = make_pytorch_oxe_iterable_dataset(dataset, train=True, batch_size=512, transform_dict=cfg.transforms)
    for i, sample in tqdm.tqdm(enumerate(dataloader)):
        print("Top-level keys: ", sample.keys())
        print("Observation keys: ", sample["observation"].keys())
        print("Task keys: ", sample["task"].keys())
        break


if __name__ == "__main__":
    main()
