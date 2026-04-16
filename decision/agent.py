from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from decision.env import PortfolioEnv

def create_agent():
    env = make_vec_env(lambda: PortfolioEnv(), n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
    )

    return model, env
