import os
import time
import torch
import numpy as np
import wandb
from tqdm import tqdm

# Local imports
from config import Config
from agent import PPOAgent
from env import create_rlgym_env_factory, TorchVecNormalize
from utils import set_seed, log_gpu_usage, evaluate_agent


def main():
    """Main training function."""
    cfg = Config()
    run_name = f"{cfg.EXP_NAME}_{cfg.SEED}_{int(time.time())}"

    # Initialize Weights & Biases if enabled
    if cfg.WANDB_LOG:
        wandb.init(
            project=cfg.WANDB_PROJECT_NAME,
            entity=cfg.WANDB_ENTITY,
            sync_tensorboard=True,
            config=vars(cfg),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

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
    print("🏗️ Creating environments...")
    env_factories = [create_rlgym_env_factory(cfg.NUM_AGENTS_PER_ENV) for _ in range(cfg.NUM_ENVS)]
    # We no longer use AsyncVectorEnv here
    envs = TorchVecNormalize(env_factories, device, cfg)

    # --- 3. Agent Initialization ---
    print("🧠 Initializing neural network...")
    # Get observation and action space specs from the single-env space
    first_agent_key = next(iter(envs.single_observation_space.spaces))
    single_agent_obs_shape = envs.single_observation_space.spaces[first_agent_key].shape
    action_space_n = envs.single_action_space.spaces[first_agent_key].n

    # The network receives the concatenated observations of all agents.
    flat_obs_shape = (single_agent_obs_shape[0] * cfg.TOTAL_AGENTS_PER_ENV,)

    agent = PPOAgent(single_agent_obs_shape, action_space_n, device, cfg)
    trainable_params = sum(p.numel() for p in agent.policy.parameters() if p.requires_grad)
    print(f"📏 Network size: {trainable_params:,} trainable parameters")

    # --- 4. Storage Initialization ---
    print("📦 Preallocating memory on GPU...")
    # This tensor will store the flattened observations for direct indexing later.
    flat_obs_storage = torch.zeros((cfg.NUM_STEPS, cfg.NUM_ENVS) + flat_obs_shape, device=device)
    
    actions_storage = torch.zeros((cfg.NUM_STEPS, cfg.NUM_ENVS, cfg.TOTAL_AGENTS_PER_ENV), dtype=torch.long, device=device)
    logprobs_storage = torch.zeros((cfg.NUM_STEPS, cfg.NUM_ENVS, cfg.TOTAL_AGENTS_PER_ENV), device=device)
    rewards_storage = torch.zeros((cfg.NUM_STEPS, cfg.NUM_ENVS, cfg.TOTAL_AGENTS_PER_ENV), device=device)
    dones_storage = torch.zeros((cfg.NUM_STEPS, cfg.NUM_ENVS, cfg.TOTAL_AGENTS_PER_ENV), device=device)
    values_storage = torch.zeros((cfg.NUM_STEPS, cfg.NUM_ENVS, cfg.TOTAL_AGENTS_PER_ENV), device=device)

    # --- 5. Training Loop ---
    num_updates = cfg.TOTAL_TIMESTEPS // cfg.BATCH_SIZE
    global_step = 0
    best_eval_reward = -float('inf')
    start_time = time.time()

    print(f"\n🚀 Starting Training Loop for {num_updates} updates!")
    print("=" * 80)

    # Initial reset
    next_obs_dict, _ = envs.reset(seed=cfg.SEED)
    next_done = torch.zeros(cfg.NUM_ENVS, cfg.TOTAL_AGENTS_PER_ENV, device=device)
    agent_keys = list(next_obs_dict.keys())

    with tqdm(range(1, num_updates + 1), desc="Training") as pbar:
        for update in pbar:
            # --- Learning Rate Annealing ---
            if cfg.ANNEAL_LR:
                frac = 1.0 - (update - 1.0) / num_updates
                lr_now = frac * cfg.LEARNING_RATE
                agent.optimizer.param_groups[0]["lr"] = lr_now

            # --- Rollout Phase ---
            agent.policy.eval()
            for step in range(cfg.NUM_STEPS):
                global_step += cfg.NUM_ENVS * cfg.TOTAL_AGENTS_PER_ENV
                
                dones_storage[step] = next_done

                # Flatten obs dict for the policy network and store it
                flat_obs = torch.cat([next_obs_dict[key] for key in agent_keys], dim=1)
                flat_obs_storage[step] = flat_obs
                
                # Get action, logprob, and value from the agent
                with torch.no_grad():
                    action, logprob, _, value = agent.policy.get_action_and_value(flat_obs)
                    values_storage[step] = value.squeeze(-1)
                
                actions_storage[step] = action
                logprobs_storage[step] = logprob

                # Format actions for the environment step function
                actions_to_send = {}
                for i, key in enumerate(agent_keys):
                    # Reshape to (num_envs, 1) to ensure each worker receives an array, not a scalar.
                    actions_to_send[key] = action[:, i].cpu().numpy().reshape(-1, 1)
                
                # Step the environment
                next_obs_dict, rewards_tensor, next_done, _ = envs.step(actions_to_send)

                rewards_storage[step] = rewards_tensor

            # --- Advantage Calculation (GAE) ---
            agent.policy.eval()
            with torch.no_grad():
                flat_next_obs = torch.cat([next_obs_dict[key] for key in agent_keys], dim=1)
                next_value = agent.policy.get_value(flat_next_obs).squeeze(-1) # Squeeze the last dim here
                advantages = torch.zeros_like(rewards_storage, device=device)
                lastgaelam = 0
                for t in reversed(range(cfg.NUM_STEPS)):
                    if t == cfg.NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_storage[t + 1]
                        nextvalues = values_storage[t + 1]
                    
                    delta = rewards_storage[t] + cfg.GAMMA * nextvalues * nextnonterminal - values_storage[t]
                    advantages[t] = lastgaelam = delta + cfg.GAMMA * cfg.GAE_LAMBDA * nextnonterminal * lastgaelam
                returns = advantages + values_storage

            # --- Flatten and Prepare Batches ---
            b_obs = flat_obs_storage.reshape((-1,) + flat_obs_shape)
            b_logprobs = logprobs_storage.reshape(-1, cfg.TOTAL_AGENTS_PER_ENV)
            b_actions = actions_storage.reshape(-1, cfg.TOTAL_AGENTS_PER_ENV)
            b_advantages = advantages.reshape(-1, cfg.TOTAL_AGENTS_PER_ENV)
            b_returns = returns.reshape(-1, cfg.TOTAL_AGENTS_PER_ENV)
            b_values = values_storage.reshape(-1, cfg.TOTAL_AGENTS_PER_ENV)
            
            # --- Training Phase ---
            agent.policy.train()
            clipfracs = []
            for _ in range(cfg.NUM_UPDATE_EPOCHS):
                b_inds = np.random.permutation(cfg.BATCH_SIZE)
                for start in range(0, cfg.BATCH_SIZE, cfg.MINIBATCH_SIZE):
                    end = start + cfg.MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]

                    pg_loss, v_loss, entropy_loss, _, _, clipfrac = agent.update(
                        b_obs[mb_inds],
                        b_actions[mb_inds],
                        b_logprobs[mb_inds],
                        b_advantages[mb_inds],
                        b_returns[mb_inds],
                        b_values[mb_inds]
                    )
                    clipfracs.append(clipfrac)

            # --- Logging ---
            sps = int(global_step / (time.time() - start_time))
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            # Update the tqdm progress bar with live metrics
            pbar.set_postfix(
                SPS=sps,
                VLoss=f"{v_loss.item():.2f}",
                PLoss=f"{pg_loss.item():.2f}",
                GPU=log_gpu_usage(device)
            )

            if cfg.WANDB_LOG:
                wandb.log({
                    "global_step": global_step,
                    "SPS": sps,
                    "learning_rate": agent.optimizer.param_groups[0]['lr'],
                    "value_loss": v_loss.item(),
                    "policy_loss": pg_loss.item(),
                    "entropy": entropy_loss.item(),
                    "explained_variance": explained_var,
                    "clipfrac": np.mean(clipfracs),
                })

            # --- Evaluation and Model Saving ---
            if (update % cfg.EVAL_INTERVAL == 0 or update == num_updates) and cfg.SAVE_MODEL:
                pbar.write(f"\n🧪 Running evaluation at update {update}...")
                eval_env_factory = create_rlgym_env_factory(cfg.NUM_AGENTS_PER_ENV)
                eval_reward = evaluate_agent(eval_env_factory, agent, device, cfg.EVAL_EPISODES, cfg)
                
                if cfg.WANDB_LOG:
                    wandb.log({"charts/eval_mean_reward": eval_reward, "global_step": global_step})

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    model_path = f"models/{run_name}/best_model.pt"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save(agent.policy.state_dict(), model_path)
                    pbar.write(f"💾 New best model saved! Reward: {best_eval_reward:.2f} -> {model_path}")
                else:
                    pbar.write(f"📊 Eval reward: {eval_reward:.2f} (best: {best_eval_reward:.2f})")

    # --- Final Cleanup ---
    envs.close()
    if cfg.WANDB_LOG:
        wandb.finish()
    print("=" * 80)
    print("🏆 Training Finished!")
    print("=" * 80)


if __name__ == '__main__':
    main()
