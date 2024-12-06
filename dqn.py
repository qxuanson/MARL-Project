import random
import torch
import torch.nn as nn
import torch.optim as optim
from math import exp
from collections import namedtuple
from torch_model import QNetwork
from magent2.environments import battle_v4

RED = "red"
BLUE = "blue"

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 25000
TAU = 0.005
LR = 1e-4
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class ReplayBuffer:
    def __init__(self, capacity):
        self.__list = []
        self.__max_size = capacity
        self.__cursor = 0

    def push(self, *args):
        if len(self.__list) < self.__max_size:
            self.__list.append(Transition(*args))
        else: 
            self.__list[self.__cursor] = Transition(*args)
        self.__cursor = (self.__cursor + 1) % self.__max_size

    def sample(self, batch_size = 1):
        return random.sample(self.__list, batch_size)

    def __len__(self):
        return len(self.__list)
    

env = battle_v4.env(map_size=45, render_mode="rgb_array", max_cycles=1000)

observation_shape = env.observation_space("red_0").shape
action_shape = env.action_space("red_0").n

policy_net = QNetwork(observation_shape, action_shape).to(device)
target_net = QNetwork(observation_shape, action_shape).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayBuffer(10000)

steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.item()
    else:
        return env.action_space(agent).sample()
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    non_final_mask = torch.tensor([not d for d in batch.done], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d])

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 10
for i_episode in range(num_episodes):
    print("episode " + str(i_episode + 1))
    env.reset()
    states = {}
    actions = {}
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, _ = env.last()

        done = termination or truncation
        if agent[:3] == RED:
            next_state = (torch.tensor(observation, device=device).float().permute([2, 0, 1]).unsqueeze(0))
            if states.get(agent, None) is not None:
                memory.push(states[agent], actions[agent], next_state, torch.tensor([reward], device=device), done)

            if done: 
                action = None
            else: 
                action = select_action(next_state)
                states[agent] = next_state
                actions[agent] = torch.tensor([[action]], device=device, dtype=torch.long)

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        else:
            if done: 
                action = None
            else: 
                action = env.action_space(agent).sample()
        
        env.step(action)

    save_path = 'weights/model_weights' + str(i_episode + 1) + '.pth'
    torch.save(policy_net.state_dict(), save_path)
    print("Mô hình đã được lưu vào " + save_path)








