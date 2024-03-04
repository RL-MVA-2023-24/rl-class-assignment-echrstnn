from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 4),
        )

    def act(self, observation, use_random=False):
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            act_val = self(observation).squeeze()
            act = act_val.argmax().item()
            print(f"{act}  {act_val.numpy()}")
            return act

    def forward(self, observation):
        x = torch.log10(observation)
        x = self.layers(x)
        return torch.exp(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self):
        print("Loading...")
        self.load_state_dict(torch.load("model.pt"))
