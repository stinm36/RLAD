import gym
import d4rl
import torch

from torch.utils.data import Dataset


class D4RLDataset(Dataset):
    def __init__(self, env_name):
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        self.observations = dataset['observations']
        self.action = dataset['actions']
        self.data = torch.cat([
            torch.tensor(self.observations, dtype=torch.float32),
            torch.tensor(self.action, dtype=torch.float32)
        ], dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class D4RLDataset_Partial(Dataset):
    """D4RLDataset의 부분 데이터셋 버전. 미리 필터링된 dataset dict를 받음."""
    def __init__(self, dataset):
        self.observations = dataset['observations']
        self.action = dataset['actions']
        self.data = torch.cat([
            torch.tensor(self.observations, dtype=torch.float32),
            torch.tensor(self.action, dtype=torch.float32)
        ], dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
