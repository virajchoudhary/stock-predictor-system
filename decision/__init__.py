import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PortfolioEnv(gym.Env):

    def __init__(self, n_assets=3, steps=200):
        super().__init__()

        self.n_assets = n_assets
        self.steps = steps
        self.current_step = 0

        # Observation: asset returns
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(n_assets,), dtype=np.float32
        )

        # Action: portfolio weights
        self.action_space = spaces.Box(
            low=0, high=1, shape=(n_assets,), dtype=np.float32
        )

        # Synthetic returns
        self.returns = np.random.normal(0.001, 0.02, (steps, n_assets))

        # Initialize previous allocation (equal weight)
        self.prev_action = np.ones(self.n_assets) / self.n_assets

    def reset(self, seed=None, options=None):
        self.current_step = 0

        # Reset previous allocation each episode
        self.prev_action = np.ones(self.n_assets) / self.n_assets

        return self.returns[self.current_step], {}

    def step(self, action):

        # Safe normalization
        total = np.sum(action)
        if total <= 1e-8:
            action = np.ones(self.n_assets) / self.n_assets
        else:
            action = action / total

        portfolio_return = np.dot(action, self.returns[self.current_step])

        # Risk penalty
        risk = portfolio_return ** 2
        lambda_risk = 5

        # Transaction cost penalty
        transaction_cost = np.sum(np.abs(action - self.prev_action))
        eta_cost = 0.1

        reward = portfolio_return - lambda_risk * risk - eta_cost * transaction_cost

        # Update previous action
        self.prev_action = action.copy()

        self.current_step += 1
        done = self.current_step >= self.steps - 1

        next_state = self.returns[self.current_step]

        return next_state, reward, done, False, {}