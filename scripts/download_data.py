# minimum working example to download a single OXE dataset
from uha import download_oxe_data
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="uha.data.utils", config_name="uha_default_load_config")
def main(cfg: DictConfig):
    download_oxe_data(cfg)


if __name__ == "__main__":
    main()
