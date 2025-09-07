import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import NoTouchTimeoutCondition, AnyCondition
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward
from rlgym_tools.rocket_league.reward_functions.demo_reward import DemoReward
from rlgym_tools.rocket_league.reward_functions.episode_end_reward import EpisodeEndReward
from rlgym_tools.rocket_league.reward_functions.flip_reset_reward import FlipResetReward
from rlgym_tools.rocket_league.reward_functions.goal_prob_reward import GoalViewReward
from rlgym_tools.rocket_league.reward_functions.wavedash_reward import WavedashReward
from rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward
from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import AerialDistanceReward
from rlgym_tools.rocket_league.reward_functions.boost_change_reward import BoostChangeReward
from rlgym_tools.rocket_league.reward_functions.boost_keep_reward import BoostKeepReward
from rlgym_tools.rocket_league.reward_functions.ball_travel_reward import BallTravelReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym_ppo.util import RLGymV2GymWrapper
from rlgym_tools.rocket_league.obs_builders.relative_default_obs import RelativeDefaultObs
from rlgym_tools.rocket_league.state_mutators.game_mutator import GameMutator
from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardProvider
from rlgym_tools.rocket_league.done_conditions.game_condition import GameCondition

from rewards import (
    RecoveryReward,
    DoubleTapReward,
    WallPlayReward,
    RotationReward,
    OpponentPressureReward,
    ChallengeReward,
    BallProximityReward,
    TimeoutPenaltyReward,
    VelocityBallToGoalReward,
)
from renderer import RocketSimVisRenderer
from conditions import NoGoalTimeoutCondition


def build_rlgym_env(render_mode=None, spawn_opponents=True, stage=1):
    """
    Builds the RLGym environment with a reward structure based on the training stage.
    """
    team_size = 1
    action_repeat = 8

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)

    # Base rewards present in all stages
    base_rewards = [
        (GoalReward(), 100.0)
    ]

    # Stage 1: The absolute fundamentals of offense
    stage1_rewards = base_rewards + [
        (AdvancedTouchReward(), 2.0),
        (VelocityPlayerToBallReward(
            include_negative_values=True,
            use_trajectory_comparison=True,
            use_dot_quotient=True
        ), 1.0),
        (GoalViewReward(), 0.5),
        (BallProximityReward(), 0.01),
    ]

    # Stage 2: Build on Stage 1 by adding defense and game sense
    stage2_rewards = stage1_rewards + [
        (DemoReward(), 20.0),
        (VelocityBallToGoalReward(), 2.0),
        (BallTravelReward(goal_weight=2.0), 1.0),
        (TimeoutPenaltyReward(), -100),
    ]

    # Stage 3: Build on Stage 2 by adding advanced mechanics
    stage3_rewards = stage2_rewards + [
        (EpisodeEndReward(), 10.0),
        (FlipResetReward(), 50.0),
        (WavedashReward(), 8.0),
        (AerialDistanceReward(), 20.0),
        (BoostChangeReward(), 5.0),
        (BoostKeepReward(), 10.0),
        (DoubleTapReward(), 75.0),
        (WallPlayReward(), 15.0),
        (RecoveryReward(), 25.0),
        (OpponentPressureReward(), 15.0),
        (RotationReward(), 12.0),
        (ChallengeReward(), 10.0),
    ]
    
    reward_fn = None
    if render_mode is None:
        if stage == 1:
            print("... Building Stage 1 Environment: Foundational Offense ...")
            reward_fn = CombinedReward(*stage1_rewards)
        elif stage == 2:
            print("... Building Stage 2 Environment: Adding Game Sense & Defense ...")
            reward_fn = CombinedReward(*stage2_rewards)
        elif stage >= 3:
            print("... Building Stage 3 Environment: Full Mechanics ...")
            reward_fn = CombinedReward(*stage3_rewards)
        else:
             reward_fn = GoalReward()
    else:
        reward_fn = GoalReward()

    obs_builder = RelativeDefaultObs()
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size if spawn_opponents else 0),
        KickoffMutator(),
        GameMutator()
    )
    shared_info_provider = ScoreboardProvider()
    termination_condition = GameCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=20.0),
        NoGoalTimeoutCondition(timeout_seconds=60.0),
    )
    renderer = RocketSimVisRenderer() if render_mode == "human" else None

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        shared_info_provider=shared_info_provider,
        transition_engine=RocketSimEngine(),
        truncation_cond=truncation_condition,
        renderer=renderer
    )

    return RLGymV2GymWrapper(rlgym_env)
