import numpy as np
from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from env_hiv import HIVPatient
from gymnasium.wrappers.time_limit import TimeLimit
from train import ProjectAgent
import math
from itertools import count
from tqdm import tqdm

torch.set_num_threads(16)

raw_env = HIVPatient(domain_randomization=True)
env = TimeLimit(env=raw_env, max_episode_steps=200)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.01
EPS_END = 0.005
EPS_DECAY = 200 * 10
TAU = 0.005
LR = 1e-5
device = "cuda"
num_episodes = 10000
reward_rescale = 1e-8

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = ProjectAgent()
policy_net.load()
policy_net.to(device)
target_net = ProjectAgent().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(200 * 50)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # print(loss)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()


for i_episode in range(num_episodes):
    raw_env.domain_randomization = i_episode % 5 != 0
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0
    episode_loss = 0

    for t in tqdm(count(), total=200, disable=True):
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward *= reward_rescale
        episode_reward += reward
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        episode_loss += optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[
                key
            ] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break

    episode_reward /= reward_rescale
    episode_loss /= 200
    print(f"Episode {i_episode}  reward={episode_reward:.3e}  loss={episode_loss:.3e}")
    if i_episode % 20 == 19:
        policy_net.cpu()
        policy_net.save(f"model-{i_episode}.pt")
        policy_net.to(device)

print("Complete")
