# Snake Game with Reinforcement Learning

A modular implementation of the classic Snake game with reinforcement learning using Gymnasium and Stable Baselines3.

## Project Structure

```
├── snake_game.py      # Core game logic + manual play
├── snake_env.py       # Gymnasium environment wrapper
├── train_snake.py     # Training script with SB3
├── pyproject.toml     # Project configuration and dependencies
└── README.md          # This file
```

## Installation with uv

1. Initialize the project:
```bash
uv init
```

2. Install dependencies:
```bash
uv sync
```

## Usage

### Play Snake Manually

Before training the AI, you can play the game yourself:

```bash
# Basic play
uv run python snake_game.py

# Custom settings
uv run python snake_game.py --width 25 --height 20 --speed 10
```

**Manual Controls:**
- **Arrow Keys**: Move snake (Up, Right, Down, Left)
- **SPACE**: Pause/Unpause game
- **R**: Restart game
- **ESC**: Quit game

### Basic Training

Train with default settings (PPO, 100k steps):
```bash
uv run python train_snake.py
```

### Advanced Training Options

Train for more steps:
```bash
uv run python train_snake.py train 200000
```

Train with visual rendering (slower but you can watch):
```bash
uv run python train_snake.py train 50000 --render
```

### Watch Trained Model

Watch the AI play after training:
```bash
uv run python train_snake.py play
```

Watch specific model play 10 games:
```bash
uv run python train_snake.py play snake_model 10
```

## Environment Details

### State Space (11 dimensions)
- **Danger Detection (3)**: Straight ahead, right turn, left turn
- **Direction (4)**: Current direction (up, right, down, left)
- **Food Location (4)**: Food relative to head (up, down, left, right)

### Action Space (4 actions)
- **0**: Up
- **1**: Right
- **2**: Down
- **3**: Left

### Reward System
- **+10**: Eating food
- **-10**: Game over (collision)
- **Penalty**: Taking too long without eating (encourages food-seeking behavior)

## Files Explained

### `snake_game.py`
Pure Python implementation of Snake game logic:
- No RL dependencies
- Handles game state, collisions, scoring
- Pygame rendering support
- Clean separation of game logic from AI

### `snake_env.py`
Gymnasium environment wrapper:
- Implements standard Gym interface
- Handles observation/action space definitions
- Manages rendering modes
- Environment registration

### `train_snake.py`
Complete training pipeline:
- Algorithm support: PPO (Proximal Policy Optimization)
- Training progress monitoring
- Model saving/loading
- Command-line interface

## Expected Performance

- **Random Agent**: ~0-2 score
- **Trained Agent (50k steps)**: ~5-15 score
- **Well-trained Agent (200k+ steps)**: ~15-30+ score

