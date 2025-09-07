class Config:
    """
    Configuration class for PPO training hyperparameters.
    """
    def __init__(self):
        self.SEED = 42
        self.EXP_NAME = "RLGym_PPO_SSL3" # Base name for the experiment
        
        # --- WandB Logging ---
        self.WANDB_LOG = True 
        self.WANDB_PROJECT_NAME = "rlgym-ppo-ssl-curriculum-3"

        # --- System ---
        self.USE_MPS = True 
        self.USE_AMP = True 

        # --- PPO Hyperparameters (can be overridden by stages) ---
        self.GAMMA = 0.99 
        self.GAE_LAMBDA = 0.95 
        self.CLIP_COEF = 0.2 
        self.ENT_COEF = 0.01 
        self.VF_COEF = 0.5 
        self.MAX_GRAD_NORM = 0.5 
        self.NORM_ADV = True 
        self.CLIP_VLOSS = True 

        # --- MODIFIED: Environment & Training Defaults for Faster Learning ---
        self.NUM_ENVS = 64
        self.NUM_STEPS = 2048
        self.ANNEAL_LR = True 
        self.NUM_UPDATE_EPOCHS = 8
        self.NUM_MINIBATCHES = 4
        self.SAVE_MODEL = True
        
        # --- Calculated values (do not change) ---
        self.BATCH_SIZE = self.NUM_ENVS * self.NUM_STEPS
        self.MINIBATCH_SIZE = self.BATCH_SIZE // self.NUM_MINIBATCHES

        # --- CURRICULUM LEARNING STAGES ---
        # The policy/critic sizes are kept consistent across all stages
        # to ensure seamless loading from one stage to the next.
        self.STAGES = {
            1: {
                "EXP_NAME_SUFFIX": "Stage1_Touch",
                "TOTAL_TIMESTEPS": 1_000_000_000,
                "LEARNING_RATE": 3e-4,
                "POLICY_LAYER_SIZES": [1024, 1024, 512],
                "CRITIC_LAYER_SIZES": [1024, 1024, 512],
            },
            2: {
                "EXP_NAME_SUFFIX": "Stage2_Dribble",
                "TOTAL_TIMESTEPS": 2_000_000_000,
                "LEARNING_RATE": 1e-4,
                "POLICY_LAYER_SIZES": [1024, 1024, 512],
                "CRITIC_LAYER_SIZES": [1024, 1024, 512],
            },
            3: {
                "EXP_NAME_SUFFIX": "Stage3_FullMechanics",
                "TOTAL_TIMESTEPS": 5_000_000_000,
                "LEARNING_RATE": 5e-5,
                "POLICY_LAYER_SIZES": [1024, 1024, 512],
                "CRITIC_LAYER_SIZES": [1024, 1024, 512],
            }
        }
