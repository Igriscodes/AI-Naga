# Snake AI - Deep Reinforcement Learning

This repository contains a high-performance Snake Game that's AI trained using Deep Q-Learning (DQN). It supports both CPU-based training and massive GPU-accelerated parallel training (100+ simultaneous games each second).

## Structure

*   `train_gpu.py`: **(Recommended)** Runs parallel games on the GPU for ultra-fast training.
*   `train.py`: Traditional CPU-based training (Single instance).
*   `play.py`: Visualizes the trained agent playing the game.
*   `game.py`: The standard Pygame environment.
*   `vec_game.py`: The PyTorch-accelerated GPU environment.
*   `model.py`: The Deep Q-Network (Linear_QNet) architecture.
*   `agent.py`: The RL Agent logic.

## Requirements

*   Python 3.8+
*   PyTorch (with CUDA for GPU mode)
*   Pygame
*   NumPy
*   Matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Agent

**Option A: GPU Training (Fastest)**
This runs 10,000 games in parallel.
```bash
python train_gpu.py
```
*   The model will be saved to `./model/model_gpu.pth`.
*   Rename this to `./model/model.pth` to use it in Play Mode.

**Option B: CPU Training (Standard)**
Runs a single headless instance on the CPU.
```bash
python train.py
```



### 2. Watch the Agent Play

After training, verify the model exists at `./model/model.pth`.
Then run:
```bash
python play.py
```
![game](https://github.com/user-attachments/assets/53d4b7e8-01d4-4a3f-8243-ddabd27ef4ca)

## License
[GNU Lesser General Public License v2.1](LICENSE) - Feel free to use and modify
