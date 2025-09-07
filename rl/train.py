import torch
import argparse
from tqdm import tqdm
from rlgym_ppo import Learner
from functools import partial

from config import Config
from env import build_rlgym_env
from utils import find_latest_file_path


def main():
    parser = argparse.ArgumentParser(description="Stage-based training for RLGym PPO.")
    parser.add_argument("--stage", type=int, default=1, help="The training stage to run (e.g., 1, 2, 3).")
    args = parser.parse_args()
    
    stage_to_run = args.stage
    
    cfg = Config()
    
    # --- Get Stage-Specific Configuration ---
    stage_config = cfg.STAGES.get(stage_to_run)
    if not stage_config:
        print(f"❌ Error: Stage {stage_to_run} is not defined in config.py. Exiting.")
        return

    # --- Apply Stage-Specific Settings ---
    exp_name = f"{cfg.EXP_NAME}__{stage_config['EXP_NAME_SUFFIX']}"
    total_timesteps = stage_config['TOTAL_TIMESTEPS']
    learning_rate = stage_config['LEARNING_RATE']
    policy_layers = stage_config['POLICY_LAYER_SIZES']
    critic_layers = stage_config['CRITIC_LAYER_SIZES']
    
    print("="*80)
    print(f"🚀 STARTING TRAINING STAGE {stage_to_run}: {stage_config['EXP_NAME_SUFFIX']}")
    print(f"   Total Timesteps for this stage: {total_timesteps:,}")
    print(f"   Learning Rate: {learning_rate}")
    print("="*80)

    # Use a partial function to pass the stage argument to the environment builder
    env_builder_fn = partial(build_rlgym_env, stage=stage_to_run)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and cfg.USE_MPS else "cpu"

    # --- Initialize the Learner ---
    learner = Learner(
        env_builder_fn,
        n_proc=cfg.NUM_ENVS,
        min_inference_size=max(1, cfg.NUM_ENVS // 2),
        ppo_batch_size=cfg.BATCH_SIZE,
        ts_per_iteration=cfg.BATCH_SIZE,
        exp_buffer_size=3 * cfg.BATCH_SIZE,
        ppo_minibatch_size=cfg.MINIBATCH_SIZE,
        ppo_epochs=cfg.NUM_UPDATE_EPOCHS,
        policy_lr=learning_rate,
        critic_lr=learning_rate,
        ppo_ent_coef=cfg.ENT_COEF,
        save_every_ts=min(1_000_000, total_timesteps),
        timestep_limit=total_timesteps,
        log_to_wandb=cfg.WANDB_LOG,
        wandb_project_name=cfg.WANDB_PROJECT_NAME,
        policy_layer_sizes=policy_layers,
        critic_layer_sizes=critic_layers,
        checkpoints_save_folder=f"models_{exp_name}",
        device=device
    )

    # --- Load Model from Previous Stage (if applicable) ---
    if stage_to_run > 1:
        try:
            prev_stage_config = cfg.STAGES[stage_to_run - 1]
            prev_exp_base_name = f"{cfg.EXP_NAME}__{prev_stage_config['EXP_NAME_SUFFIX']}"
            
            model_path = find_latest_file_path(prev_exp_base_name, "PPO_POLICY.pt")
            
            # Important: Load weights *before* starting the training process
            learner.agent.policy.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Successfully loaded weights from Stage {stage_to_run - 1} model.")
        
        except Exception as e:
            print(f"⚠️ Warning: Could not load model from previous stage: {e}")
            print(f"Starting Stage {stage_to_run} from scratch.")

    # --- Run Training ---
    # The learner will handle its own internal looping and saving.
    learner.learn()
    print("\n✅ Training complete for this stage.")

if __name__ == "__main__":
    main()
