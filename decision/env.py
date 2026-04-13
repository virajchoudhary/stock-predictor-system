import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, n_assets=3, steps=200):
        super().__init__()

        self.n_assets = n_assets
        self.steps = steps
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(n_assets,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0, high=1, shape=(n_assets,), dtype=np.float32
        )

        self.returns = np.random.normal(0.001, 0.02, (steps, n_assets))

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self.returns[self.current_step], {}

    def step(self, action):
        total = np.sum(action)

        if total <= 1e-8:
            action = np.ones(self.n_assets) / self.n_assets
        else:
            action = action / total

        portfolio_return = np.dot(action, self.returns[self.current_step])

# Risk penalty (variance proxy)
        risk = portfolio_return ** 2
        lambda_risk = 5  # risk aversion parameter (tune later)
        reward = portfolio_return - lambda_risk * risk

        self.current_step += 1
        done = self.current_step >= self.steps - 1

        next_state = self.returns[self.current_step]

        return next_state, reward, done, False, {}