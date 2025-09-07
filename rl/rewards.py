import numpy as np
from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values


class DoubleTapReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards potential double tap setups and completions.
    This logic is global (not per-agent) as it only tracks ball state, 
    matching the provided example's logic.
    """
    def __init__(self):
        self.double_tap_sequence = False
        self.first_touch_height = 0.0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.double_tap_sequence = False
        self.first_touch_height = 0.0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        reward_val = 0.0
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        
        # Detect double tap setup (high ball velocity towards a backboard)
        if ball_pos[2] > 400 and abs(ball_vel[1]) > 1000:
            if not self.double_tap_sequence:
                self.double_tap_sequence = True
                self.first_touch_height = ball_pos[2]
                reward_val += 0.1  # Setup reward

        # Detect first touch (high velocity touch while ball is high)
        if (self.double_tap_sequence and
            ball_pos[2] > self.first_touch_height * 0.8 and
            np.linalg.norm(ball_vel) > 1500):
            reward_val += 0.3  # First touch reward

        # Detect second touch (goal-bound high-velocity touch)
        if (self.double_tap_sequence and
            ball_pos[2] > 200 and
            abs(ball_vel[1]) > 2000): # Assuming this implies shot velocity
            reward_val += 0.6  # Second touch (completion) reward
            self.double_tap_sequence = False
        
        # Reset sequence if ball hits ground
        if ball_pos[2] < 100:
            self.double_tap_sequence = False
            
        return {agent: reward_val for agent in agents}


class WallPlayReward(RewardFunction[AgentID, GameState, float]):
    """Rewards agent for interacting with the ball near the wall and taking wall shots."""
    def __init__(self):
        self.wall_time = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.wall_time = {agent: 0 for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        ball_pos = state.ball.position
        
        for agent in agents:
            car = state.cars[agent]
            player_pos = car.physics.position
            reward_val = 0.0
            is_near_wall = abs(player_pos[0]) > (common_values.SIDE_WALL_X - 200) # Check if near side walls
            
            # Wall proximity reward
            if is_near_wall:
                self.wall_time[agent] += 1
                reward_val += 0.1 * (self.wall_time[agent] / 50.0)
            else:
                self.wall_time[agent] = 0
                
            # Wall-ball interaction reward
            if is_near_wall and np.linalg.norm(ball_pos - player_pos) < 500:
                reward_val += 0.5
                
            # Wall shot reward (on wall, ball is high and near a goal)
            if (is_near_wall and 
                ball_pos[2] > 200 and
                abs(ball_pos[1]) > (common_values.BACK_NET_Y - 500)):
                reward_val += 0.8
                
            rewards[agent] = reward_val
            
        return rewards


class RecoveryReward(RewardFunction[AgentID, GameState, float]):
    """Rewards agent for landing correctly and penalizes for being turtle/off-ground too long."""
    def __init__(self):
        self.recovery_time = {}
        self.last_ground_contact = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.recovery_time = {agent: 0 for agent in agents}
        self.last_ground_contact = {agent: True for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        
        for agent in agents:
            car = state.cars[agent]
            player_vel = car.physics.linear_velocity
            on_ground = car.on_ground
            reward_val = 0.0
            
            # Ground contact reward
            if on_ground and not self.last_ground_contact.get(agent, True):
                reward_val += 0.5  # Landing reward
                self.recovery_time[agent] = 0
            
            # Recovery time penalty
            if not on_ground:
                self.recovery_time[agent] += 1
                if self.recovery_time[agent] > 100:  # Too long in air/off-wall without landing
                    reward_val -= 0.1
                    
            # Speed recovery reward (landed and moving fast)
            if on_ground and np.linalg.norm(player_vel) > 1000:
                reward_val += 0.3
                
            self.last_ground_contact[agent] = on_ground
            rewards[agent] = reward_val
            
        return rewards


class RotationReward(RewardFunction[AgentID, GameState, float]):
    """Rewards agent for rotating away from the ball (presumably to a defensive position)."""
    def __init__(self):
        self.last_ball_distance = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_ball_distance = {}
        for agent in agents:
            player_pos = initial_state.cars[agent].physics.position
            ball_pos = initial_state.ball.position
            self.last_ball_distance[agent] = np.linalg.norm(ball_pos - player_pos)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        ball_pos = state.ball.position
        
        for agent in agents:
            player_pos = state.cars[agent].physics.position
            ball_distance = np.linalg.norm(ball_pos - player_pos)
            
            # Rotation away from ball reward
            if ball_distance > self.last_ball_distance.get(agent, 0.0):  # Moving away from ball
                rewards[agent] = 0.2
            else:
                rewards[agent] = 0.0
                
            self.last_ball_distance[agent] = ball_distance
            
        return rewards


class OpponentPressureReward(RewardFunction[AgentID, GameState, float]):
    """Rewards agent for being close to opponents (applying pressure)."""
    def __init__(self):
        pass

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        
        for agent in agents:
            player_car = state.cars.get(agent)
            if not player_car:
                rewards[agent] = 0.0
                continue

            player_pos = player_car.physics.position
            reward_val = 0.0
            
            opponents = [c for c in state.cars.values() if c.team_num != player_car.team_num]
            
            if opponents:
                closest_opponent_dist = min(np.linalg.norm(c.physics.position - player_pos) for c in opponents)
                
                # Pressure reward
                if closest_opponent_dist < 1000:
                    reward_val += 0.3
                    
                # Challenge reward (very close)
                if closest_opponent_dist < 500:
                    reward_val += 0.5
                    
            rewards[agent] = reward_val
            
        return rewards


class ChallengeReward(RewardFunction[AgentID, GameState, float]):
    """Rewards agent for challenging the ball at high speed."""
    def __init__(self):
        pass

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        ball_pos = state.ball.position
        
        for agent in agents:
            car = state.cars[agent]
            player_pos = car.physics.position
            player_vel = car.physics.linear_velocity
            reward_val = 0.0
            
            # Challenge timing reward
            ball_distance = np.linalg.norm(ball_pos - player_pos)
            if 200 < ball_distance < 800:  # Good challenge range
                reward_val += 0.4
                
            # Speed challenge reward
            if np.linalg.norm(player_vel) > 1500:  # Fast challenge
                reward_val += 0.3
                
            rewards[agent] = reward_val
            
        return rewards


class BallProximityReward(RewardFunction[AgentID, GameState, float]):
    """
    A simple reward function that encourages the agent to be close to the ball.
    This is less prone to hacking than velocity-based rewards in a random environment.
    """
    def __init__(self):
        super().__init__()
        # Calculate the max distance on the field (3D diagonal of the arena)
        self.max_dist = np.sqrt((2 * common_values.SIDE_WALL_X)**2 + (2 * common_values.BACK_WALL_Y)**2 + common_values.CEILING_Z**2)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            dist = np.linalg.norm(car.physics.position - state.ball.position)
            # Reward is higher when closer. The reward is 1 when on the ball and 0 at max field distance.
            # Use the calculated max distance instead of a hardcoded value.
            reward = 1.0 - (dist / self.max_dist)
            rewards[agent] = np.power(reward, 2) # Square it to make being very close much more rewarding
        return rewards


class TimeoutPenaltyReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, penalty: float = 1.0):
        super().__init__()
        self.penalty = penalty

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent_id, truncated in is_truncated.items():
            if truncated:
                rewards[agent_id] = self.penalty
            else:
                rewards[agent_id] = 0.0
        return rewards


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for hitting the ball towards the opponent's goal.
    """
    def __init__(self):
        super().__init__()

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            
            # Determine opponent's goal position
            opponent_goal_y = common_values.BACK_WALL_Y if car.is_blue else -common_values.BACK_WALL_Y
            goal_pos = np.array([0, opponent_goal_y, 0])

            # Calculate vector from ball to opponent's goal
            ball_to_goal = goal_pos - state.ball.position
            
            # Project ball velocity onto the ball-to-goal vector
            velocity_towards_goal = np.dot(state.ball.linear_velocity, ball_to_goal / np.linalg.norm(ball_to_goal))
            
            # Normalize the reward
            reward = velocity_towards_goal / common_values.BALL_MAX_SPEED
            rewards[agent] = max(0, reward)  # Only give positive rewards for moving towards goal
        
        return rewards
