from decision.agent import create_agent

def train_model():
    print("Training started...")
    
    model, env = create_agent()
    
    model.learn(total_timesteps=20000)
    
    model.save("ppo_portfolio_model")
    
    print("Training completed and model saved.")

if __name__ == "__main__":
    train_model()