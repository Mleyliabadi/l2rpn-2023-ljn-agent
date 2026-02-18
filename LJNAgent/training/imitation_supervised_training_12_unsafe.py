# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

from collections import Counter

import grid2op
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from curriculumagent.teacher.submodule.encoded_action import \
    EncodedTopologyAction
from dispatcher_agent.gym_assets.action_space import GlobalTopoActionSpace
from grid2op.gym_compat import BoxGymObsSpace, GymEnv
from lightsim2grid.lightSimBackend import LightSimBackend
from stable_baselines3 import PPO
from torch.utils.data.dataset import Dataset, random_split
from tqdm import tqdm

## The list of attributes to keep in the observation ##
# TIPS : It is possible to get good results by keeping only rho vector
DEFAULT_OBS_ATTR_TO_KEEP = [
    "day_of_week",
    "hour_of_day",
    "minute_of_hour",
    "prod_p",
    "load_p",
    "actual_dispatch",
    "target_dispatch",
    "topo_vect",
    "time_before_cooldown_line",
    "time_before_cooldown_sub",
    "rho",
    "timestep_overflow",
    "storage_power",
    "storage_charge",
]


class ExpertDataSet(Dataset):
    """Torch Dataset Class for supervised training

    Parameters
    ----------
    Dataset : torch.Dataset
        Should be instantiated from collected teacher experience as a array of observation and vector actions
        as well as an action list to retrieve index for labeling.
    """

    def __init__(self, expert_observations, expert_actions, action_list):
        self.observations = expert_observations
        self.action_space = action_list
        self.actions = [self.action_space.index(act) for act in tqdm(expert_actions)]

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)


def acc_topk(action_prediction, labels, k=1):
    """Top-K accuracy

    Parameters
    ----------
    action_prediction : torch.tensor
        predicted action (as index from the action space)
    labels : torch.tensor
        ground truth labels (as index from the action space)
    k : int, optional
        Number of action to be accounted for top-k accuracy, by default 1 (standard accuracy in multiclass problem)

    Returns
    -------
    float
        top-k accuracy for the given prediction vs ground truth batch
    """
    return th.sum(
        th.eq(
            labels.reshape((len(labels), 1)), th.topk(action_prediction, k=k, dim=1)[1]
        ).any(dim=1)
    ) / len(labels)


def decode_act(act, env):
    """
    Decoding function based on the curriculum_agent encoding method - See curriculum agent repo for more info.
    """
    return EncodedTopologyAction.decode_action(act, env)


def instantiate_gym_env(env, g2op_action_list):
    """Instantiation function for the gym env for supervised training.

    Parameters
    ----------
    env : grid2op.Env
        Corresponding grid2o environment. Provided environment should match the training data.
    g2op_action_list : list
        The list of grid2op actions to be considered for the supervised training.

    Returns
    -------
    gym.Env
        Gym Env instance for training.
        This is mainly useful for compatibility for further training with Reinforcement Learning algorithms
        And it is also motivated by the convenience of the gymnasium package for RL related problems.
    """
    gym_env = GymEnv(env)
    gym_env.observation_space.close()
    gym_env.observation_space = BoxGymObsSpace(
        env.observation_space, attr_to_keep=DEFAULT_OBS_ATTR_TO_KEEP
    )
    gym_env.action_space.close()
    gym_env.action_space = GlobalTopoActionSpace(g2op_action_list, env.action_space)

    return gym_env


def prepare_data(env, dataset_path):
    """Data preparation pipeline.
    The function is responsible for generating a proper dataset from collected vector experience data.
    It instantiate the dataset, processes it and perform random split for train and test set.

    Parameters
    ----------
    env : g2op.Env
        The grid2op env. Here it should match "l2rpn_idf_2023" unless you are trying to train on another env.
    dataset_path : str
        Path to the collected supervised experience. It should be a npz file containing pairs of observation and actions as vectors.

    Returns
    -------
    (torch.Dataset, torch.Dataset)
        The randomly splitted train and test set ready for training.
    """
    data = np.load(dataset_path)
    encoded_action_list = list(Counter(data["best_action"]).keys())
    g2op_action_list = [decode_act(act, env) for act in tqdm(encoded_action_list)]

    gym_env = instantiate_gym_env(env, g2op_action_list)

    obs_vect = np.zeros((len(data["obs_old"]), gym_env.observation_space.shape[0]))
    for i, row in tqdm(enumerate(data["obs_old"])):
        g2op_obs = env.observation_space.from_vect(row)
        obs_vect[i] = gym_env.observation_space.to_gym(g2op_obs)

    dataset = ExpertDataSet(obs_vect, data["best_action"], encoded_action_list)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
        dataset, [train_size, test_size]
    )
    return gym_env, train_expert_dataset, test_expert_dataset


def train_model(
    gym_env, train_dataset, test_dataset, epoch=30, batch_size=16, num_workers=4
):
    """Simple training function for supervised learning.

    Parameters
    ----------
    gym_env : gym.Env
        Instance of the gym env. It should match the training data.
    train_dataset : torch.dataset
        Training subset
    test_dataset : torch.dataset
        Test subset
    epoch : int, optional
        Number of training epoch, by default 30
    batch_size : int, optional
        Batch size, by default 16
    num_workers : int, optional
        Number of processes, by default 4
    """
    model_rl = PPO(
        "MlpPolicy",
        gym_env,
        verbose=1,
        policy_kwargs={"net_arch": dict(pi=[1024, 1024], vf=[1024, 1024])},
    )
    model = model_rl.policy.to("cuda:0")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=0.05)

    trainloader = th.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = th.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    for epoch in tqdm(range(30)):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        train_acc = {"top_1": [], "top_5": [], "top_10": [], "top_20": []}
        training_loss = []
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            dist = model.get_distribution(inputs)
            action_prediction = dist.distribution.logits
            target = labels.long()
            loss = criterion(action_prediction, labels)
            loss.backward()
            optimizer.step()

            train_acc["top_1"].append(
                acc_topk(action_prediction, labels).cpu().numpy()
            ),
            train_acc["top_5"].append(
                acc_topk(action_prediction, labels, k=5).cpu().numpy()
            )
            train_acc["top_10"].append(
                acc_topk(action_prediction, labels, k=10).cpu().numpy()
            )
            train_acc["top_20"].append(
                acc_topk(action_prediction, labels, k=20).cpu().numpy()
            )

            # print statistics
            running_loss += loss.item()
            if i % 10000 == 999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
        training_loss.append(running_loss)
        test_acc = {"top_1": [], "top_5": [], "top_10": [], "top_20": []}
        for i, (inputs, labels) in enumerate(testloader, 0):
            inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")
            dist = model.get_distribution(inputs)
            action_prediction = dist.distribution.logits
            target = labels.long()

            test_acc["top_1"].append(acc_topk(action_prediction, labels).cpu().numpy()),
            test_acc["top_5"].append(
                acc_topk(action_prediction, labels, k=5).cpu().numpy()
            )
            test_acc["top_10"].append(
                acc_topk(action_prediction, labels, k=10).cpu().numpy()
            )
            test_acc["top_20"].append(
                acc_topk(action_prediction, labels, k=20).cpu().numpy()
            )
        print(
            f"Test Accuracy top 1: {np.mean(test_acc['top_1'])}   | Train Accuracy : {np.mean(train_acc['top_1'])}"
        )
        print(
            f"Test Accuracy top 5: {np.mean(test_acc['top_5'])}   | Train Accuracy : {np.mean(train_acc['top_5'])}"
        )
        print(
            f"Test Accuracy top 10: {np.mean(test_acc['top_10'])} | Train Accuracy : {np.mean(train_acc['top_10'])}"
        )
        print(
            f"Test Accuracy top 20: {np.mean(test_acc['top_20'])} | Train Accuracy : {np.mean(train_acc['top_20'])}"
        )

    model_rl.policy = model
    model_rl.save("12_unsafe_policy")
    print("Finished Training")


if __name__ == "__main__":
    dataset_path = "/home/jules/RTE/dataset/teacher2_label_cleaned_100_GLOBAL.npz"
    env = grid2op.make("l2rpn_idf_2023", backend=LightSimBackend())
    gym_env, training_dataset, test_dataset = prepare_data(env, dataset_path)
    train_model(gym_env, training_dataset, test_dataset)
