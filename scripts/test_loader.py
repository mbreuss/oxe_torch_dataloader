# minimum working example to load the OXE dataset
from uha import make_pytorch_oxe_iterable_dataset, get_octo_dataset_tensorflow, get_single_dataset_tensorflow
import tqdm
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../uha/data/conf", config_name="uha_default_load_config")
def main(cfg: DictConfig):
    # load Training Dataset from TensorflowDatasets
    dataset = get_octo_dataset_tensorflow(cfg, train=True)
    # dataset = get_single_dataset_tensorflow(cfg, train=True)
    is_single_dataset = False
    batch_size = 512
    cfg_transforms = OmegaConf.to_object(cfg.transforms)
    language_encoder = hydra.utils.instantiate(cfg.language_encoders)
    # create Pytorch Train Dataset
    dataloader = make_pytorch_oxe_iterable_dataset(dataset, train=True, batch_size=batch_size, transform_dict=cfg_transforms, num_workers=0, pin_memory=True, language_encoder=language_encoder, is_single_dataset=is_single_dataset, main_process=True)
    # dataloader = make_pytorch_oxe_iterable_dataset(dataset, train=True, batch_size=512)
    for sample in dataloader:
    # for i, sample in tqdm.tqdm(enumerate(dataloader)):
        print("Top-level keys: ", sample.keys())
        # print("rgb_obs keys: ", sample["rgb_obs"].keys())
        # print("Task keys: ", sample["task"].keys())
        print("action: ", sample["action"][:, 0, -1])

        # print("Top-level keys: ", sample.keys())
        # print("Task keys: ", sample["task"].keys())
        # print("task image_primary shape: ", sample["task"]["image_primary"].shape)
        # print("observation image_primary shape: ", sample["observation"]["image_primary"].shape)

        print("task language tokens: ", sample["task"]["language_instruction"]["input_ids"].shape)
        # print("observation_proprio shape: ", sample["observation"]["proprio"].shape)
        break


if __name__ == "__main__":
    main()
