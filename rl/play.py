import torch
import numpy as np
import time
import json
import argparse
from rlgym_ppo.ppo.discrete_policy import DiscreteFF

# Local imports
from env import build_rlgym_env
from utils import set_seed
from config import Config
from utils import find_latest_file_path


def main():
    """Main function to load the model and run the bot."""
    # --- Argument Parsing for specifying the stage ---
    parser = argparse.ArgumentParser(description="Run a trained RLGym PPO agent.")
    parser.add_argument("--stage", type=int, default=1, help="The training stage of the model to load.")
    args = parser.parse_args()

    cfg = Config()

    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and cfg.USE_MPS else "cpu")
    print(f"🎮 Using device: {device}")

    print(f"🏗️  Creating environment for playing (Stage {args.stage})...")
    # --- Pass the stage argument to the environment builder ---
    eval_env = build_rlgym_env(render_mode="human", spawn_opponents=True, stage=args.stage)

    try:
        # --- Find the model directory for the specified stage ---
        stage_config = cfg.STAGES.get(args.stage)
        if not stage_config:
            raise ValueError(f"Stage {args.stage} not found in config.py")

        exp_name = f"{cfg.EXP_NAME}__{stage_config['EXP_NAME_SUFFIX']}"

        model_path = find_latest_file_path(exp_name, "PPO_POLICY.pt")
        bookkeeping_path = find_latest_file_path(exp_name, "BOOK_KEEPING_VARS.json")

        print(f"🧠 Loading latest model from {model_path}...")
        print(f"📖 Reading runtime stats from {bookkeeping_path}...")

        with open(bookkeeping_path, 'r') as f:
            bookkeeping_data = json.load(f)

        obs_size = bookkeeping_data["obs_running_stats"]["shape"][0]
        act_size = eval_env.action_space.n
        
        policy_layer_sizes = stage_config["POLICY_LAYER_SIZES"]
        print(f"🧠 Using architecture from config for Stage {args.stage}: {policy_layer_sizes}")

        policy = DiscreteFF(obs_size, act_size, policy_layer_sizes, device)

        model_state_dict = torch.load(model_path, map_location=device)

        policy.load_state_dict(model_state_dict)
        policy.eval()

        print("✅ Successfully loaded model.")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        eval_env.close()
        return

    print("\n🚀 Starting bot play!")
    print("Press the 'x' button on the window or Ctrl+C in the terminal to stop.")

    SIMULATION_FPS = 15
    SIMULATION_TIME_STEP = 1 / SIMULATION_FPS
    last_sim_time = time.perf_counter()

    try:
        obs = eval_env.reset()

        while True:
            current_time = time.perf_counter()
            if current_time - last_sim_time >= SIMULATION_TIME_STEP:
                last_sim_time = current_time

                with torch.no_grad():
                    if obs.ndim == 1:
                        obs = np.expand_dims(obs, axis=0)

                    agent_actions = []
                    for agent_obs in obs:
                        obs_tensor = torch.from_numpy(agent_obs).float().to(device)
                        obs_tensor = obs_tensor.unsqueeze(0)

                        action_tensor, _ = policy.get_action(obs_tensor, deterministic=True)
                        
                        action = action_tensor.squeeze().item()
                        agent_actions.append([action])

                    actions_to_send = np.array(agent_actions).reshape(-1, 1)
                                            
                    obs, _, terminated, truncated, _ = eval_env.step(actions_to_send)
                    
                    if terminated or truncated:
                        obs = eval_env.reset()

            eval_env.render()

    except KeyboardInterrupt:
        print("\n✅ User interrupted. Stopping.")
    finally:
        eval_env.close()
        print("✅ Environment closed.")


if __name__ == '__main__':
    main()
