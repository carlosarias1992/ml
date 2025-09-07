import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# --- 1. Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, single_obs_shape, num_actions, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.single_agent_obs_size = int(np.prod(single_obs_shape))

        # Shared body for feature extraction
        self.shared_body = nn.Sequential(
            nn.Linear(self.single_agent_obs_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        # Critic head (estimates state value)
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Actor head (outputs action logits)
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, num_actions)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, np.sqrt(2))
            module.bias.data.zero_()

    def get_value(self, x):
        """
        Gets the value of a state.
        Input x has shape (batch_size, num_agents * single_obs_size).
        """
        batch_size = x.shape[0]
        # Reshape to (batch_size * num_agents, single_obs_size) to process each agent's obs
        x_reshaped = x.view(batch_size * self.num_agents, self.single_agent_obs_size)

        features = self.shared_body(x_reshaped)
        values = self.critic_head(features) # Shape: (batch_size * num_agents, 1)

        # Reshape back to (batch_size, num_agents, 1)
        return values.view(batch_size, self.num_agents, 1)


    def get_action_and_value(self, x, action=None):
        """
        Gets actions and values for a batch of states.
        Input x has shape (batch_size, num_agents * single_obs_size).
        """
        batch_size = x.shape[0]
        # Reshape to (batch_size * num_agents, single_obs_size)
        x_reshaped = x.view(batch_size * self.num_agents, self.single_agent_obs_size)

        features = self.shared_body(x_reshaped)
        logits = self.actor_head(features) # Shape: (batch_size * num_agents, num_actions)
        values = self.critic_head(features) # Shape: (batch_size * num_agents, 1)

        # Reshape logits and values to be per-agent
        logits_reshaped = logits.view(batch_size, self.num_agents, -1)
        values_reshaped = values.view(batch_size, self.num_agents, 1)

        # Create a distribution over actions
        probs = Categorical(logits=logits_reshaped)
        
        if action is None:
            action = probs.sample() # Shape: (batch_size, num_agents)
        
        log_prob = probs.log_prob(action) # Shape: (batch_size, num_agents)
        entropy = probs.entropy() # Shape: (batch_size, num_agents)

        return action, log_prob, entropy, values_reshaped

# --- 2. PPO Agent ---
class PPOAgent:
    def __init__(self, single_obs_shape, num_actions, device, cfg):
        self.device = device
        self.cfg = cfg
        # Pass num_agents to the policy network
        self.policy = ActorCritic(single_obs_shape, num_actions, cfg.TOTAL_AGENTS_PER_ENV).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.LEARNING_RATE, eps=1e-5)

    def update(self, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values):
        """Performs a PPO update step."""
        # Get new logprobs, entropy, and values from the policy
        _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(
            b_obs, b_actions
        )
        
        logratio = newlogprob.flatten() - b_logprobs.flatten()
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.cfg.CLIP_COEF).float().mean().item()

        mb_advantages = b_advantages.flatten()
        if self.cfg.NORMALIZE_ADVANTAGES:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.CLIP_COEF, 1 + self.cfg.CLIP_COEF)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if self.cfg.CLIP_COEF: # Assuming CLIP_COEF is intended to also act as a flag
            v_loss_unclipped = (newvalue - b_returns.flatten()) ** 2
            v_clipped = b_values.flatten() + torch.clamp(
                newvalue - b_values.flatten(),
                -self.cfg.CLIP_COEF,
                self.cfg.CLIP_COEF,
            )
            v_loss_clipped = (v_clipped - b_returns.flatten()) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns.flatten()) ** 2).mean()

        # Entropy loss
        entropy_loss = entropy.mean()
        
        # Total loss
        loss = pg_loss - self.cfg.ENT_COEF * entropy_loss + v_loss * self.cfg.VF_COEF

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.MAX_GRAD_NORM)
        self.optimizer.step()
        
        return pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs
