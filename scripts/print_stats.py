# minimum working example to load the OXE Stats and print the transitions / trajectories
from uha.data.oxe import make_oxe_dataset_kwargs_and_weights
import hydra
from omegaconf import DictConfig
import tensorflow_datasets as tfds
import tensorflow as tf
import dlimp as dl
import inspect

from uha.data.utils.data_utils import get_dataset_statistics


@hydra.main(config_path="../uha/data/conf", config_name="uha_default_load_config")
def main(cfg: DictConfig):
    # load Training Dataset from TensorflowDatasets
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.DATA_PATH,
        load_camera_views=cfg.load_camera_views,
    )

    results = []
    num_transitions = 0
    num_trajectories = 0
    for dataset in dataset_kwargs_list:
        name = dataset["name"]
        data_dir = dataset["data_dir"]
        builder = tfds.builder(name, data_dir=data_dir)

        full_dataset = dl.DLataset.from_rlds(
            builder, split="all", shuffle=False
        )
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(dataset["state_obs_keys"]),
                inspect.getsource(dataset["standardize_fn"]) if dataset["standardize_fn"] is not None else "",
            ),
            save_dir=builder.data_dir,
        )
        num_transitions += dataset_statistics["num_transitions"]
        num_trajectories += dataset_statistics["num_trajectories"]
        results.append({"name": name, "num_trajectories": dataset_statistics["num_trajectories"], "num_transitions": dataset_statistics["num_transitions"]})
        # dataset_statistics = tree_map(np.array, dataset_statistics)

    print("transitions contained in data: ", num_transitions)
    print("trajectories contained in data: ", num_trajectories)
    print(results)


if __name__ == "__main__":
    main()
