import torch
import torch.nn as nn
import torch.optim as optim
import time
from vec_game import VectorSnakeGameGPU
from model import Linear_QNet
import os

NUM_ENVS = 10000 
LR = 0.001
GAMMA = 0.9

def train():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. GPU training impossible.")
        return

    print(f"Initializing parallel environments on GPU...")
    env = VectorSnakeGameGPU(num_envs=NUM_ENVS, device='cuda')
    
    model = Linear_QNet(11, 256, 3).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print("Training Started. Watch the 'Games/Sec' counter!")
    print("-------------------------------------------------")
    
    state = env.reset() 
    
    total_steps = 0
    start_time = time.time()
    
    epsilon = 1.0
    
    try:
        while True:
            total_steps += 1
            
            if epsilon > 0.01:
                epsilon *= 0.9995
            
            random_mask = torch.rand(NUM_ENVS, device='cuda') < epsilon
            
            pred = model(state) 
            
            final_actions = torch.zeros((NUM_ENVS, 3), device='cuda')
            
            model_choices = torch.argmax(pred, dim=1)
            final_actions[range(NUM_ENVS), model_choices] = 1
            
            random_choices = torch.randint(0, 3, (NUM_ENVS,), device='cuda')
            final_actions[random_mask] = 0
            final_actions[random_mask, random_choices[random_mask]] = 1
            
            next_state, reward, done, scores = env.step(final_actions)
            
            with torch.no_grad():
                next_pred = model(next_state)
                target_q = reward + GAMMA * torch.max(next_pred, dim=1)[0] * (~done)
            
            action_indices = torch.argmax(final_actions, dim=1)
            current_q = pred.gather(1, action_indices.unsqueeze(1)).squeeze()
            
            loss = criterion(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            
            if total_steps % 100 == 0:
                duration = time.time() - start_time
                games_per_sec = (NUM_ENVS * 100) / duration
                avg_score = scores.mean().item()
                max_score = scores.max().item()
                
                print(f"Steps: {total_steps} | Max Score: {max_score:.0f} | Avg Score: {avg_score:.2f} | Speed: {games_per_sec:.0f} moves/sec")
                
                start_time = time.time()
                
                if max_score > 50:
                    print(">>> High Score Reached! Saving Model.")
                    model.save('model_gpu.pth')

    except KeyboardInterrupt:
        print("Stopped.")
        model.save('model_gpu.pth')

if __name__ == '__main__':
    train()