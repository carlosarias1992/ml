class Config:
    # --- Experiment ---
    EXP_NAME = "RLGym_PPO_Refactored"
    SEED = 42
    TORCH_DETERMINISTIC = True
    USE_MPS = True # Use Apple Silicon GPU if available

    # --- Environment ---
    NUM_ENVS = 32  # Number of parallel environments
    NUM_AGENTS_PER_ENV = 2 # e.g., 1v1 would be 2 agents
    TOTAL_AGENTS_PER_ENV = NUM_AGENTS_PER_ENV # For clarity in storage calculations

    # --- PPO ---
    TOTAL_TIMESTEPS = 10_000_000
    NUM_STEPS = 512  # Steps per environment per policy rollout
    ANNEAL_LR = True
    LEARNING_RATE = 2.5e-4
    NUM_UPDATE_EPOCHS = 4
    NUM_MINIBATCHES = 4
    UPDATE_BATCH_SIZE = NUM_ENVS * NUM_STEPS
    MINIBATCH_SIZE = UPDATE_BATCH_SIZE // NUM_MINIBATCHES
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_COEF = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    NORMALIZE_ADVANTAGES = True
    TARGET_KL = None # 0.015 is a good starting value if you want to use it

    # --- Network ---
    POLICY_HIDDEN_LAYERS = [256, 256]
    VALUE_HIDDEN_LAYERS = [256, 256]

    # --- Logging & Saving ---
    SAVE_MODEL = True
    EVAL_INTERVAL = 50  # Number of updates between evaluations
    EVAL_EPISODES = 10 # Number of episodes to run for each evaluation

    # Weights & Biases
    WANDB_LOG = True
    WANDB_PROJECT_NAME = "rlgym-ppo"
    WANDB_ENTITY = None # wandb entity name (e.g., username)

    # --- Computed values ---
    BATCH_SIZE = int(NUM_ENVS * NUM_STEPS)
    TOTAL_AGENTS = NUM_ENVS * TOTAL_AGENTS_PER_ENV
