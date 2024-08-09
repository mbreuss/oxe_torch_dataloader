# minimum working example to load the OXE Stats and print the transitions / trajectories
from uha.data.oxe import make_oxe_dataset_kwargs_and_weights
import hydra
from omegaconf import DictConfig
import tensorflow_datasets as tfds
import tensorflow as tf
import dlimp as dl
import inspect
from uha.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from uha.data.utils.spec import ModuleSpec
from uha.data.utils.data_utils import sample_match_keys_uniform

from uha.data.utils.data_utils import get_dataset_statistics


@hydra.main(config_path="../uha/data/conf", config_name="uha_default_load_config")
def main(cfg: DictConfig):
    # load Training Dataset from TensorflowDatasets
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.DATA_PATH,
        load_camera_views=cfg.load_camera_views,
    )

    REQUIRED_KEYS = {"observation", "action"}
    # name = "bridge"
    proprio_obs_key = None
    language_key = None

    def restructure(traj):
        # apply a standardization function, if provided
        if standardize_fn is not None:
            traj = ModuleSpec.instantiate(standardize_fn)(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. "
                "Did you write a `standardize_fn`?"
            )

        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}

        if proprio_obs_key is not None:
            new_obs["proprio"] = tf.cast(old_obs[proprio_obs_key], tf.float32)

        # add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # extracts `language_key` into the "task" dict, or samples uniformly if `language_key` fnmatches multiple keys
        task = {}
        if language_key is not None:
            task["language_instruction"] = sample_match_keys_uniform(traj, language_key)
            if task["language_instruction"].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {task['language_instruction'].dtype}, "
                    "but it must be tf.string."
                )

        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

        return traj

    def is_nonzero_length(traj):
        return tf.shape(traj["action"])[0] > 0

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
        standardize_fn =  ModuleSpec.create(OXE_STANDARDIZATION_TRANSFORMS[name])
        full_dataset = full_dataset.traj_map(restructure).filter(is_nonzero_length)
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                # str(dataset["state_obs_keys"]),
                # inspect.getsource(dataset["standardize_fn"]) if dataset["standardize_fn"] is not None else "",
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
