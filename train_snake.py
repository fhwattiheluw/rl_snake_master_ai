# train_snake.py
from stable_baselines3 import PPO
from snake_env import SnakeEnv

def train_snake(timesteps=100000, render=False):
    """Train Snake AI with PPO"""

    print("Training Snake with PPO")
    print(f"Total timesteps: {timesteps}")
    print("-" * 40)

    # Create environment
    render_mode = "human" if render else None
    env = SnakeEnv(render_mode=render_mode)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=timesteps)

    # Save the model
    model.save("snake_model")
    print("Model saved as 'snake_model'")

    env.close()
    return model

def play_trained_model(model_path="snake_model", episodes=5):
    """Watch the trained model play"""

    print(f"Loading model: {model_path}")

    # Create environment with rendering
    env = SnakeEnv(render_mode="human")

    # Load model
    model = PPO.load(model_path, env=env)

    print(f"Watching trained agent play {episodes} episodes...")
    print("Close the window to stop early")

    scores = []
    for episode in range(episodes):
        obs, info = env.reset()
        done = False

        print(f"Episode {episode + 1}:")

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        score = info.get('score', 0)
        scores.append(score)
        print(f"Score: {score}")

    env.close()

    print(f"\nResults:")
    print(f"Average Score: {sum(scores)/len(scores):.2f}")
    print(f"Best Score: {max(scores)}")

    return scores

def main():
    """Main function with simple command line interface"""
    import sys

    if len(sys.argv) == 1:
        # Default training
        train_snake()
    elif sys.argv[1] == "train":
        # Custom training
        timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
        render = "--render" in sys.argv
        train_snake(timesteps, render)
    elif sys.argv[1] == "play":
        # Play trained model
        model_path = sys.argv[2] if len(sys.argv) > 2 else "snake_model"
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        play_trained_model(model_path, episodes)
    else:
        print("Usage:")
        print("  python train_snake.py                    # Train with defaults")
        print("  python train_snake.py train 200000       # Train for 200k steps")
        print("  python train_snake.py train 50000 --render # Train with rendering")
        print("  python train_snake.py play               # Watch trained model")
        print("  python train_snake.py play snake_model 3 # Watch model play 3 games")

if __name__ == "__main__":
    main()