import torch
import os
import time
from agent import Agent
from game import SnakeGameAI

def play():
    agent = Agent()
    game = SnakeGameAI()
    
    model_path = './model/model.pth'
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    else:
        print("No saved model found at ./model/model.pth. Playing with random weights.")

    agent.n_games = 1000 
    
    print("Starting Play Mode... Press Ctrl+C to stop.")

    try:
        while True:
            state_old = agent.get_state(game)

            final_move = agent.get_action(state_old)

            reward, done, score = game.play_step(final_move)
            
            if done:
                game.reset()
                print('Game Over. Score:', score)
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\nPlay mode stopped.")

if __name__ == '__main__':
    play()