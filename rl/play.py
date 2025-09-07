import torch
import numpy as np
import pygame

# Local imports
from config import Config
from agent import PPOAgent
from env import create_rlgym_env_factory, TorchVecNormalize
from utils import set_seed


def main():
    """Main function to load the model and run the bot."""
    cfg = Config()
    
    # --- 1. Setup ---
    set_seed(cfg.SEED)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 Using NVIDIA CUDA")
    elif torch.backends.mps.is_available() and cfg.USE_MPS:
        device = torch.device("mps")
        print("🍎 Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
        
    # --- 2. Environment Creation ---
    print("🏗️ Creating environment for playing...")
    # Use a single, non-vectorized environment for evaluation
    eval_env_factory = create_rlgym_env_factory(cfg.NUM_AGENTS_PER_ENV, render_mode="human")
    eval_env = eval_env_factory()
    agent_keys = list(eval_env.observation_space.spaces.keys())

    # --- 3. Agent Initialization ---
    print("🧠 Initializing neural network...")
    first_agent_key = next(iter(eval_env.observation_space.spaces))
    single_agent_obs_shape = eval_env.observation_space.spaces[first_agent_key].shape
    action_space_n = eval_env.action_space.spaces[first_agent_key].n
    
    agent = PPOAgent(single_agent_obs_shape, action_space_n, device, cfg)

    # --- 4. Load Model ---
    try:
        model_path = f"models/{cfg.EXP_NAME}_{cfg.SEED}_*/best_model.pt"
        # FIX: Glob to find the latest model in case the timestamp changes
        import glob
        latest_model = glob.glob(model_path)
        if not latest_model:
            raise FileNotFoundError("No model found. Please train the agent first.")
        
        agent.policy.load_state_dict(torch.load(latest_model[0]))
        print(f"✅ Successfully loaded model from {latest_model[0]}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        eval_env.close()
        return

    # --- 5. Play Loop ---
    print("\n🚀 Starting bot play!")
    print("Press the 'x' button on the window or Ctrl+C in the terminal to stop.")
    
    # Set the policy to evaluation mode
    agent.policy.eval()
    
    try:
        # Loop to play episodes until the user closes the window
        obs_dict, _ = eval_env.reset()
        
        while True:
            # Handle Pygame window events to allow for closing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            
            # Create a batch of observations for this single environment state
            obs_tensors = [torch.tensor(obs_dict[key], dtype=torch.float32, device=device).unsqueeze(0) for key in agent_keys]
            flat_obs = torch.cat(obs_tensors, dim=1)

            # Get actions from the policy
            with torch.no_grad():
                actions_tensor, _, _, _ = agent.policy.get_action_and_value(flat_obs)
            
            # Format actions for the environment
            actions_to_send = {key: np.array([actions_tensor[0, i].item()]) for i, key in enumerate(agent_keys)}

            # Step the environment
            obs_dict, rewards, terminated, truncated, _ = eval_env.step(actions_to_send)
            
            # Check for episode termination
            done = any(terminated.values()) or any(truncated.values())
            
            # Render the environment
            eval_env.render()
            
            # If the episode is done, reset the environment
            if done:
                obs_dict, _ = eval_env.reset()
            
    except KeyboardInterrupt:
        print("\n✅ User interrupted. Stopping.")
    finally:
        eval_env.close()
        print("✅ Environment closed.")


if __name__ == '__main__':
    main()
