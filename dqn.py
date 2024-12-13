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

class TeamTrainer:
    def __init__(self, observation_shape, action_shape):
        self.policy_net = QNetwork(observation_shape, action_shape).to(device)
        self.target_net = QNetwork(observation_shape, action_shape).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayBuffer(10000)
        
        self.steps_done = 0
        self.observation_shape = observation_shape
        self.action_shape = action_shape

    def select_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * exp(-1. * self.steps_done / EPS_DECAY)
        sample = random.random()
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.item()
        else:
            return random.randint(0, self.action_shape - 1)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor([not d for d in batch.done], device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

# Environment setup
env = battle_v4.env(map_size=45, render_mode="rgb_array", max_cycles=1000)

# Get observation and action shapes
observation_shape = env.observation_space("red_0").shape
action_shape = env.action_space("red_0").n

# Initialize team trainers
red_trainer = TeamTrainer(observation_shape, action_shape)
blue_trainer = TeamTrainer(observation_shape, action_shape)

# Training loop
num_episodes = 1000
for i_episode in range(num_episodes):
    print(f"Episode {i_episode + 1}")
    env.reset()
    
    red_states = {}
    red_actions = {}
    blue_states = {}
    blue_actions = {}
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, _ = env.last()
        done = termination or truncation

        # Prepare state tensor
        next_state = torch.tensor(observation, device=device).float().permute([2, 0, 1]).unsqueeze(0)
        
        if agent.startswith(RED):
            # Red team logic
            if red_states.get(agent, None) is not None:
                red_trainer.memory.push(
                    red_states[agent], 
                    red_actions[agent], 
                    next_state, 
                    torch.tensor([reward], device=device), 
                    done
                )

            if done:
                action = None
            else:
                action = red_trainer.select_action(next_state)
                red_states[agent] = next_state
                red_actions[agent] = torch.tensor([[action]], device=device, dtype=torch.long)

            red_trainer.optimize_model()
            red_trainer.soft_update_target_network()

        elif agent.startswith(BLUE):
            # Blue team logic
            if blue_states.get(agent, None) is not None:
                blue_trainer.memory.push(
                    blue_states[agent], 
                    blue_actions[agent], 
                    next_state, 
                    torch.tensor([reward], device=device), 
                    done
                )

            if done:
                action = None
            else:
                action = blue_trainer.select_action(next_state)
                blue_states[agent] = next_state
                blue_actions[agent] = torch.tensor([[action]], device=device, dtype=torch.long)

            blue_trainer.optimize_model()
            blue_trainer.soft_update_target_network()
        
        env.step(action)

    # Save only blue team's model weights
    blue_save_path = f'weights/blue_model_weights{i_episode + 1}.pth'
    torch.save(blue_trainer.policy_net.state_dict(), blue_save_path)
    print(f"Blue team model saved to {blue_save_path}")