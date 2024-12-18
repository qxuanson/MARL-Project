import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import random
from collections import defaultdict
from magent2.environments import battle_v4

import matplotlib.pyplot as plt

from model import EnhancedQNetwork
from replay_buffer import ReplayBuffer

class ImprovedTrainer:
    def __init__(self, env, input_shape, action_shape, learning_rate=1e-4):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = EnhancedQNetwork(input_shape, action_shape).to(self.device)
        self.target_network = EnhancedQNetwork(input_shape, action_shape).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, 
        #     T_max=400,
        #     eta_min=1e-5
        # )
        self.replay_buffer = ReplayBuffer(capacity=200000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.985
        self.update_target_every = 10
        self.batch_size = 4096
        self.tau = 0.005
        self.metrics = defaultdict(list)
        
    def select_action(self, observation, agent):
        if random.random() <= self.epsilon:
            return self.env.action_space(agent).sample()

        with torch.no_grad():
            q_values = self.q_network(observation)
            if self.epsilon > 0.1:
                noise = torch.randn_like(q_values) * 0.1
                q_values += noise
        return torch.argmax(q_values, dim=1).item()

    # def soft_update(self):
    #     with torch.no_grad():
    #         for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
    #             target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update_model(self, dataloader):
        total_loss = 0
        num_batches = 0
        
        for states, actions, rewards, next_states, dones in dataloader:
            state = states.to(self.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = next_states.to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

            # Double Q-learning
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = self.criterion(current_q_values, expected_q_values)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0
        
    def training(self, episodes=300):
        start_time = time.time()
        best_reward = float('-inf')
        window_size = 20
        no_improvement_count = 0
        
        for episode in range(episodes):
            self.env.reset()
            total_reward = 0
            
            # Check for blue agents at the start of each episode
            blue_agents = [agent for agent in self.env.agents if agent.startswith('blue')]
            
            reward_for_agent = {agent: 0 for agent in blue_agents}
            prev_observation = {}
            prev_action = {}
    
            # First phase
            for idx, agent in enumerate(self.env.agent_iter()):
                agent_handle = agent.split('_')[0]
                done = self.env.terminations[agent] or self.env.truncations[agent]
                
                if agent_handle == 'blue':
                    observation = torch.FloatTensor(np.transpose(self.env.observe(agent), (2, 0, 1))).to(self.device)
                    reward = self.env._cumulative_rewards[agent]
                    total_reward += reward
                    reward_for_agent[agent] += reward
                    if done:
                        action = None
                    else:
                        action = self.select_action(observation, agent)
                    prev_observation[agent] = observation
                    prev_action[agent] = action
                else:
                    if done:
                        action = None
                    else:
                        action = self.env.action_space(agent).sample()
    
                self.env.step(action)
                
                if (idx + 1) % self.env.num_agents == 0:
                    break
    
            # Second phase
            step = self.env.num_agents
            for agent in self.env.agent_iter():
                step += 1
                agent_handle = agent.split('_')[0]
                done = self.env.terminations[agent] or self.env.truncations[agent]
                
                if agent_handle == 'blue':
                    observation = torch.FloatTensor(np.transpose(self.env.observe(agent), (2, 0, 1))).to(self.device)
                    reward = self.env._cumulative_rewards[agent]
                    total_reward += reward
                    reward_for_agent[agent] += reward
                    if done:
                        action = None
                    else:
                        action = self.select_action(observation, agent)
                    self.replay_buffer.add(
                        prev_observation[agent],
                        prev_action[agent],
                        reward,  
                        observation,
                        done
                    )
                    prev_observation[agent] = observation
                    prev_action[agent] = action
                else:
                    if done:
                        action = None
                    else:
                        action = self.env.action_space(agent).sample()
    
                self.env.step(action)
    
            # Training update
            if len(self.replay_buffer) >= self.batch_size:
                dataloader = DataLoader(self.replay_buffer, batch_size=self.batch_size, shuffle=True)
                loss = self.update_model(dataloader)
                self.metrics['losses'].append(loss)
                # self.soft_update()
                if episode % self.update_target_every == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

            self.metrics['episodes'].append(episode)
            self.metrics['total_rewards'].append(total_reward)
            
            # Calculate metrics and update parameters
            max_reward = max(reward_for_agent.values())
            avg_reward = total_reward / len(blue_agents)
            self.metrics['avg_reward'].append(avg_reward)
            
            if len(self.metrics['total_rewards']) >= window_size:
                current_avg = np.mean(self.metrics['total_rewards'][-window_size:])
                current_std = np.std(self.metrics['total_rewards'][-window_size:])
                
                if current_avg > best_reward:
                    best_reward = current_avg
                    no_improvement_count = 0
                    torch.save(self.q_network.state_dict(), "best_model.pt")
                else:
                    no_improvement_count += 1
                    
                if no_improvement_count >= 30 and current_std < 5.0:
                    print(f"Early stopping at episode {episode}")
                    break
                
            print(f"Episode {episode}, Epsilon: {self.epsilon:.2f}, Total Reward: {total_reward}, "
                    f"Steps: {step}, Max Reward: {max_reward}, Avg Reward: {avg_reward:.2f}")
                
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            # self.scheduler.step()
        end_time = time.time()
        print(f"Execution time: {(end_time - start_time):.6f} seconds")

    def plot_metrics(self):
        plt.figure(figsize=(20, 12))
        plt.style.use('seaborn')

        # Reward
        plt.subplot(2, 3, 1)
        plt.plot(self.metrics['episodes'], self.metrics['total_rewards'], 'b-', label='Total Reward')
        plt.plot(self.metrics['episodes'], self.metrics['avg_reward'], 'r--', label='Avg Reward')
        plt.title('Rewards over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)

        #Loss
        plt.subplot(2, 3, 2)
        plt.plot(self.metrics['episodes'], self.metrics['losses'], 'g-')
        plt.title('Loss over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)
        

if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode=None)
    trainer = ImprovedTrainer(env, env.observation_space("blue_0").shape, env.action_space("blue_0").n)
    trainer.training()
    trainer.plot_metrics()