import random
import numpy as np
import torch
from tqdm import tqdm


def set_seed(seed, torch_deterministic=True):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def log_gpu_usage(device):
    """Logs the GPU memory usage."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        return f"CUDA: {allocated:.1f}MB / {reserved:.1f}MB"
    elif device.type == 'mps':
        return "MPS: Active"
    return "CPU: N/A"


def evaluate_agent(eval_env_factory, agent, device, num_episodes, cfg):
    """
    Evaluates the agent's performance over a number of episodes.
    """
    print("🚀 Starting Evaluation...")
    eval_env = eval_env_factory()
    agent.policy.eval() # Set the policy to evaluation mode

    total_rewards = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            obs_dict, _ = eval_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                # The evaluation environment is not vectorized, so we process one env at a time
                agent_keys = list(obs_dict.keys())
                
                # Create a batch of observations for this single environment state
                obs_tensors = [torch.tensor(obs_dict[key], dtype=torch.float32, device=device).unsqueeze(0) for key in agent_keys]
                flat_obs = torch.cat(obs_tensors, dim=1)

                # Get actions from the policy
                actions_tensor, _, _, _ = agent.policy.get_action_and_value(flat_obs)
                
                # Correctly format actions as single-element numpy arrays
                actions_to_send = {key: np.array([actions_tensor[0, i].item()]) for i, key in enumerate(agent_keys)}

                # Step the environment
                obs_dict, rewards, terminated, truncated, _ = eval_env.step(actions_to_send)
                
                # For evaluation, we can just track the reward of the first agent (e.g., blue)
                episode_reward += rewards[agent_keys[0]]

                # Check for episode termination
                done = any(terminated.values()) or any(truncated.values())

            total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"✅ Evaluation Complete - Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    agent.policy.train() # Set the policy back to training mode
    eval_env.close()
    
    return mean_reward
