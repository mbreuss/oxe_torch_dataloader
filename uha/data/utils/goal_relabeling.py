"""
Contains simple goal relabeling logic for BC use-cases where rewards and next_observations are not required.
Each function should add entries to the "task" dict.
"""

from typing import Optional

import tensorflow as tf

from uha.data.utils.data_utils import tree_merge


def uniform(traj: dict, max_goal_distance: Optional[int] = None) -> dict:
    """
    Relabels with a true uniform distribution over future states.
    Optionally caps goal distance.
    """
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]

    # select a random future index for each transition i in the range [i, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len), tf.float32)
    if max_goal_distance is not None:
        high = tf.cast(
            tf.minimum(tf.range(traj_len) + max_goal_distance, traj_len), tf.float32
        )
    else:
        high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # adds keys to "task" mirroring "observation" keys (must do a tree merge to combine "pad_mask_dict" from
    # "observation" and "task" properly)
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj["observation"])
    traj["task"] = tree_merge(traj["task"], goal)

    return traj


def uniform_and_future(traj: dict, max_goal_distance: Optional[int] = None, frame_diff: int = 1,) -> dict:
    """
    Relabels with a true uniform distribution over future states.
    Optionally caps goal distance.
    """
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]

    # select a random future index for each transition i in the range [i, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len), tf.float32)
    if max_goal_distance is not None:
        high = tf.cast(
            tf.minimum(tf.range(traj_len) + max_goal_distance, traj_len), tf.float32
        )
    else:
        high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # adds keys to "task" mirroring "observation" keys (must do a tree merge to combine "pad_mask_dict" from
    # "observation" and "task" properly)
    # goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj["observation"])
    goal = {
        "image_primary": tf.gather(traj["observation"]["image_primary"], goal_idxs),
        "pad_mask_dict": tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj["observation"]["pad_mask_dict"]),
        "timestep": tf.gather(traj["observation"]["timestep"], goal_idxs)
    }
    traj["task"] = tree_merge(traj["task"], goal)
    

    future_idxs = tf.range(traj_len)[:, None] + tf.range(frame_diff, 2*frame_diff+1, frame_diff)
    future_idxs = tf.minimum(future_idxs, traj_len - 1)
    
    traj["future_obs"] = {"image_primary": tf.gather(traj["observation"]["image_primary"], future_idxs)}
    # traj["future_obs"] = tf.nest.map_structure(lambda x: tf.gather(x, future_idxs), traj["observation"])

    return traj