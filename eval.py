from magent2.environments import battle_v4
import torch
import numpy as np
from model import EnhancedQNetwork
from torch_model import QNetwork
from final_torch_model import finalQNetwork
import imageio
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # Fallback: tqdm becomes a no-op


def eval():
    max_cycles = 300
    env = battle_v4.env(map_size=45, max_cycles=max_cycles, render_mode="rgb_array")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def random_policy(env, agent, obs):
        return env.action_space(agent).sample()

    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("red.pt", weights_only=True, map_location="cpu")
    )
    q_network.to(device)
    
    final_q_network = finalQNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    final_q_network.load_state_dict(
        torch.load("red_final.pt", weights_only=True, map_location="cpu")
    )
    final_q_network.to(device)


    def pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]
        
    def final_pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = final_q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]
    
    blue_model = EnhancedQNetwork(
        env.observation_space("blue_0").shape,
        env.action_space("blue_0").n,
    ).to(device)
    blue_model.load_state_dict(torch.load("best_model1.pt", weights_only=True, map_location="cpu"))

    def blue_policy(env, agent, obs):
        observation = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = blue_model(observation)
            return torch.argmax(q_values, dim=1).item()
    
    def run_eval(env, red_policy, blue_policy, n_episode: int = 100):
        red_win, blue_win = [], []
        red_tot_rw, blue_tot_rw = [], []
        n_agent_each_team = len(env.env.action_spaces) // 2

        for _ in tqdm(range(n_episode)):
            frames = []
            add_frame = True
            
            env.reset()
            n_kill = {"red": 0, "blue": 0}
            red_reward, blue_reward = 0, 0

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                agent_team = agent.split("_")[0]

                n_kill[agent_team] += (
                    reward > 4.5
                )  # This assumes default reward settups
                if agent_team == "red":
                    red_reward += reward
                else:
                    blue_reward += reward

                if termination or truncation:
                    action = None  # this agent has died
                else:
                    if agent_team == "red":
                        action = red_policy(env, agent, observation)
                    else:
                        action = blue_policy(env, agent, observation)

                env.step(action)
                
                if 'red' in agent:
                    if add_frame:
                        frames.append(env.render())
                        add_frame = False
                else:
                    add_frame = True
            frames.append(env.render())

            with imageio.get_writer('test-video.gif', mode='I', fps=48) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print("Done recording pretrained agents")

            who_wins = "red" if n_kill["red"] >= n_kill["blue"] + 5 else "draw"
            who_wins = "blue" if n_kill["red"] + 5 <= n_kill["blue"] else who_wins
            red_win.append(who_wins == "red")
            blue_win.append(who_wins == "blue")

            red_tot_rw.append(red_reward / n_agent_each_team)
            blue_tot_rw.append(blue_reward / n_agent_each_team)

        return {
            "winrate_red": np.mean(red_win),
            "winrate_blue": np.mean(blue_win),
            "average_rewards_red": np.mean(red_tot_rw),
            "average_rewards_blue": np.mean(blue_tot_rw),
        }

    print("=" * 20)
    print("Eval with random policy")
    print(
        run_eval(
            env=env, red_policy=random_policy, blue_policy=blue_policy, n_episode=30
        )
    )
    print("=" * 20)

    print("Eval with trained policy")
    print(
        run_eval(
            env=env, red_policy=pretrain_policy, blue_policy=blue_policy, n_episode=30
        )
    )
    print("=" * 20)

    print("Eval with final trained policy")
    print(
        run_eval(
            env=env,
            red_policy=final_pretrain_policy,
            blue_policy=blue_policy,
            n_episode=30,
        )
    )
    print("=" * 20)


if __name__ == "__main__":
    eval()