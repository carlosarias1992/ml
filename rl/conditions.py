from rlgym.api import AgentID, DoneCondition
from rlgym.rocket_league.api import GameState
from typing import List, Dict, Any
from rlgym.rocket_league import common_values


class NoGoalTimeoutCondition(DoneCondition[AgentID, GameState]):
    """
    A condition that ends the episode if a goal has not been scored within a specified time limit.
    """
    def __init__(self, timeout_seconds: float = 60.0):
        super().__init__()
        self.timeout_ticks = timeout_seconds * common_values.TICKS_PER_SECOND
        self._last_goal_tick = 0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        # Start the timer at the beginning of the episode
        self._last_goal_tick = initial_state.tick_count

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        # If a goal has just been scored, reset our timer
        if state.goal_scored:
            self._last_goal_tick = state.tick_count
        
        # Check if the time since the last goal has exceeded our timeout
        time_since_goal = state.tick_count - self._last_goal_tick
        is_timed_out = time_since_goal >= self.timeout_ticks
        
        # This condition applies to all agents
        return {agent: is_timed_out for agent in agents}
