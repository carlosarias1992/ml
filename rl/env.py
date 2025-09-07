import time
import gymnasium as gym
import numpy as np
import torch
import pygame
from gymnasium.spaces import Dict as DictSpace
from rlgym.api import RLGym
from rlgym.api.config import Renderer
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, AnyCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator


class PygameRenderer(Renderer):
    def __init__(self):
        self.screen_width = 800
        self.screen_height = 600
        self.scale = 35 # Scaling factor for the arena
        self.screen_center = (self.screen_width / 2, self.screen_height / 2)
        
        self.car_size = 50
        self.ball_size = 15
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("RLGym Bot Viewport")
        self.clock = pygame.time.Clock()

    def render(self, state, shared_info):
        # Clear the screen
        self.screen.fill((0, 0, 0)) # Black background

        # Draw the arena
        arena_color = (40, 40, 40)
        pygame.draw.rect(self.screen, arena_color, (
            self.screen_center[0] - 5120/2 / self.scale,
            self.screen_center[1] - 8192/2 / self.scale,
            5120 / self.scale,
            8192 / self.scale,
        ), 2)
        
        # Draw the ball
        ball_pos = state.ball.position
        pygame.draw.circle(self.screen, (255, 255, 255), (
            self.screen_center[0] + ball_pos[0] / self.scale,
            self.screen_center[1] + ball_pos[1] / self.scale
        ), self.ball_size)

        # Draw the cars
        for car_id in state.cars:
            car = state.cars[car_id]
            car_pos = car.physics.position
            
            # Get car team and assign color
            team = car.team_num
            color = (0, 0, 255) if team == 0 else (255, 0, 0)
            
            pygame.draw.rect(self.screen, color, (
                self.screen_center[0] + car_pos[0] / self.scale - self.car_size/2,
                self.screen_center[1] + car_pos[1] / self.scale - self.car_size/2,
                self.car_size,
                self.car_size
            ))
            
            # Draw a line to show the car's orientation
            orientation = car.physics._rotation_mtx[:, 0]
            start_pos = (
                self.screen_center[0] + car_pos[0] / self.scale,
                self.screen_center[1] + car_pos[1] / self.scale
            )
            end_pos = (
                self.screen_center[0] + car_pos[0] / self.scale + orientation[0] * self.car_size / 2,
                self.screen_center[1] + car_pos[1] / self.scale + orientation[1] * self.car_size / 2
            )
            pygame.draw.line(self.screen, (255, 255, 255), start_pos, end_pos, 3)

        pygame.display.flip()

    def close(self):
        pygame.quit()


def create_rlgym_env_factory(num_agents_per_env: int, render_mode=None, render_fps=60):
    """
    Returns a function that creates an instance of the RLGym environment.
    """
    def _init():
        # Determine team sizes based on the number of agents
        blue_size = num_agents_per_env // 2
        orange_size = num_agents_per_env - blue_size

        state_mutator = MutatorSequence(
            FixedTeamSizeMutator(blue_size=blue_size, orange_size=orange_size),
            KickoffMutator()
        )
        obs_builder = DefaultObs()
        lookup_table_action = LookupTableAction()
        action_parser = RepeatAction(lookup_table_action)
        
        reward_fn = CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 0.1)
        )
        termination_cond = GoalCondition()
        truncation_cond = AnyCondition(
            TimeoutCondition(timeout_seconds=300.),
            NoTouchTimeoutCondition(timeout_seconds=20.)
        )
        
        # We now correctly create the PygameRenderer instance here
        if render_mode == "human":
            renderer = PygameRenderer()
        else:
            renderer = None
            
        transition_engine = RocketSimEngine()

        # The renderer instance is passed directly to the RLGym constructor
        env = RLGym(
            state_mutator=state_mutator,
            obs_builder=obs_builder,
            action_parser=action_parser,
            reward_fn=reward_fn,
            termination_cond=termination_cond,
            truncation_cond=truncation_cond,
            transition_engine=transition_engine,
            renderer=renderer,
        )

        # Make the reset method compliant with Gymnasium's API
        original_reset = env.reset
        def gymnasium_compliant_reset(seed=None, options=None):
            obs_dict = original_reset()
            return obs_dict, {}
        env.reset = gymnasium_compliant_reset

        # Wrap the step method to be Gymnasium compliant
        original_step = env.step
        def gymnasium_compliant_step(actions):
            obs_dict, reward_dict, terminated_dict, truncated_dict = original_step(actions)
            # Add an empty info dictionary to match the Gymnasium API
            info = {}
            return obs_dict, reward_dict, terminated_dict, truncated_dict, info
        env.step = gymnasium_compliant_step
        
        # Ensure the environment is properly initialized
        obs_dict, _ = None, None
        while obs_dict is None or not obs_dict:
            try:
                obs_dict, _ = env.reset()
            except Exception as e:
                time.sleep(0.5)

        # Define observation and action spaces based on the first agent
        agent_ids = list(obs_dict.keys())
        first_agent_id = agent_ids[0]

        obs_shape = obs_dict[first_agent_id].shape
        single_obs_space = gym.spaces.Box(-np.inf, np.inf, obs_shape, dtype=np.float32)

        num_actions = len(lookup_table_action._lookup_table)
        single_action_space = gym.spaces.Discrete(num_actions)

        env.observation_space = DictSpace({agent_id: single_obs_space for agent_id in agent_ids})
        env.action_space = DictSpace({agent_id: single_action_space for agent_id in agent_ids})

        env.metadata = {'render_modes': ['human']}
        env.render_mode = render_mode

        return env
    return _init


class TorchVecNormalize:
    def __init__(self, env_factories, device, cfg):
        self.device = device
        self.cfg = cfg
        self.envs = [env_factory() for env_factory in env_factories]
        
        # Get obs and action space from the first env
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        
        # Get agent keys from the first environment
        self.agent_keys = list(self.envs[0].observation_space.spaces.keys())

        # Initialize running mean and variance on the specified device
        obs_shape = self.single_observation_space.spaces[self.agent_keys[0]].shape
        self.running_mean = torch.zeros(obs_shape, dtype=torch.float32, device=device)
        self.running_var = torch.ones(obs_shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(1e-8, dtype=torch.float32, device=device)

    def _update_running_stats(self, obs_batch: torch.Tensor):
        batch_mean = torch.mean(obs_batch, dim=0)
        batch_var = torch.var(obs_batch, dim=0, unbiased=False)
        batch_count = obs_batch.shape[0]

        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count

        new_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.running_mean = new_mean
        self.running_var = new_var
        self.count = tot_count

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        normalized_obs = (obs - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
        return torch.clamp(normalized_obs, -10.0, 10.0)

    def step(self, actions_to_send):
        obs_dicts = [None] * self.cfg.NUM_ENVS
        rewards_tensors = [None] * self.cfg.NUM_ENVS
        dones_tensors = [None] * self.cfg.NUM_ENVS

        for i in range(self.cfg.NUM_ENVS):
            action_dict_for_env = {key: actions_to_send[key][i] for key in self.agent_keys}
            
            obs_dict, reward_dict, terminated_dict, truncated_dict, _ = self.envs[i].step(action_dict_for_env)

            obs_dicts[i] = obs_dict
            
            # Flatten multi-agent rewards and dones for this single environment
            rewards_tensor = torch.tensor([reward_dict[k] for k in self.agent_keys], dtype=torch.float32, device=self.device)
            dones_tensor = torch.tensor([terminated_dict[k] or truncated_dict[k] for k in self.agent_keys], dtype=torch.float32, device=self.device)

            rewards_tensors[i] = rewards_tensor
            dones_tensors[i] = dones_tensor

        # Collect all agent observations into a single list before stacking
        all_obs_list = []
        for i in range(self.cfg.NUM_ENVS):
            for key in self.agent_keys:
                all_obs_list.append(obs_dicts[i][key])
        
        all_obs_batch_raw = np.stack(all_obs_list, axis=0)
        all_obs_batch = torch.tensor(all_obs_batch_raw, dtype=torch.float32, device=self.device)
        
        self._update_running_stats(all_obs_batch)
        normalized_batch = self._normalize(all_obs_batch)

        # Split the batch back into individual environment-specific obs dictionaries
        split_sizes_per_env = [self.cfg.TOTAL_AGENTS_PER_ENV for _ in range(self.cfg.NUM_ENVS)]
        split_obs = torch.split(normalized_batch, split_sizes_per_env)
        
        # Create the final output dictionary
        final_obs_dict = {}
        for key_idx, key in enumerate(self.agent_keys):
            final_obs_dict[key] = torch.cat([split_obs[i][key_idx].unsqueeze(0) for i in range(self.cfg.NUM_ENVS)], dim=0)

        # Aggregate rewards and dones
        final_rewards_tensor = torch.stack(rewards_tensors, dim=0)
        final_dones_tensor = torch.stack(dones_tensors, dim=0)

        return final_obs_dict, final_rewards_tensor, final_dones_tensor, {}


    def reset(self, **kwargs):
        obs_dicts = [self.envs[i].reset()[0] for i in range(self.cfg.NUM_ENVS)]
        
        # Collect all agent observations into a single list before stacking
        all_obs_list = []
        for i in range(self.cfg.NUM_ENVS):
            for key in self.agent_keys:
                all_obs_list.append(obs_dicts[i][key])

        all_obs_batch_raw = np.stack(all_obs_list, axis=0)
        all_obs_batch = torch.tensor(all_obs_batch_raw, dtype=torch.float32, device=self.device)
        
        self._update_running_stats(all_obs_batch)
        normalized_batch = self._normalize(all_obs_batch)

        # Split the batch back into individual environment-specific obs dictionaries
        split_sizes_per_env = [self.cfg.TOTAL_AGENTS_PER_ENV for _ in range(self.cfg.NUM_ENVS)]
        split_obs = torch.split(normalized_batch, split_sizes_per_env)
        
        # Create the final output dictionary
        final_obs_dict = {}
        for key_idx, key in enumerate(self.agent_keys):
            final_obs_dict[key] = torch.cat([split_obs[i][key_idx].unsqueeze(0) for i in range(self.cfg.NUM_ENVS)], dim=0)

        return final_obs_dict, {}

    def close(self):
        for env in self.envs:
            env.close()
