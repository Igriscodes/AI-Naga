import torch
import numpy as np

class VectorSnakeGameGPU:
    def __init__(self, num_envs=1000, width=640, height=480, block_size=20, device='cuda'):
        self.num_envs = num_envs
        self.w = width
        self.h = height
        self.block_size = block_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.grid_w = self.w // self.block_size
        self.grid_h = self.h // self.block_size
        
        self.heads = torch.zeros((num_envs, 2), device=self.device, dtype=torch.long)
        self.foods = torch.zeros((num_envs, 2), device=self.device, dtype=torch.long)
        self.directions = torch.zeros((num_envs, 2), device=self.device, dtype=torch.long)
        
        self.scores = torch.zeros(num_envs, device=self.device, dtype=torch.float)
        self.dones = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        
        self.reset()

    def reset(self):
        self.heads[:, 0] = self.grid_w // 2
        self.heads[:, 1] = self.grid_h // 2
        
        self.directions[:] = 0
        self.directions[:, 0] = 1 
        
        self.foods[:, 0] = torch.randint(0, self.grid_w, (self.num_envs,), device=self.device)
        self.foods[:, 1] = torch.randint(0, self.grid_h, (self.num_envs,), device=self.device)
        
        self.scores[:] = 0
        self.dones[:] = False
        
        return self.get_state()

    def step(self, actions):
        action_indices = torch.argmax(actions, dim=1)
        
        right_mask = (action_indices == 1)
        new_dir_x = torch.where(right_mask, -self.directions[:, 1], self.directions[:, 0])
        new_dir_y = torch.where(right_mask, self.directions[:, 0], self.directions[:, 1])
        
        left_mask = (action_indices == 2)
        new_dir_x = torch.where(left_mask, self.directions[:, 1], new_dir_x)
        new_dir_y = torch.where(left_mask, -self.directions[:, 0], new_dir_y)
        
        self.directions[:, 0] = new_dir_x
        self.directions[:, 1] = new_dir_y
        
        self.heads += self.directions
        
        c1 = self.heads[:, 0] < 0
        c2 = self.heads[:, 0] >= self.grid_w
        c3 = self.heads[:, 1] < 0
        c4 = self.heads[:, 1] >= self.grid_h
        
        crashed = c1 | c2 | c3 | c4
        
        rewards = torch.zeros(self.num_envs, device=self.device)
        rewards[crashed] = -10
        
        ate_food = (self.heads[:, 0] == self.foods[:, 0]) & (self.heads[:, 1] == self.foods[:, 1])
        rewards[ate_food] = 10
        self.scores[ate_food] += 1
        
        if ate_food.any():
            num_eaten = ate_food.sum().item()
            self.foods[ate_food, 0] = torch.randint(0, self.grid_w, (num_eaten,), device=self.device)
            self.foods[ate_food, 1] = torch.randint(0, self.grid_h, (num_eaten,), device=self.device)

        if crashed.any():
            self.heads[crashed, 0] = self.grid_w // 2
            self.heads[crashed, 1] = self.grid_h // 2
            self.scores[crashed] = 0
            self.directions[crashed, 0] = 1
            self.directions[crashed, 1] = 0

        self.dones = crashed
        
        return self.get_state(), rewards, self.dones, self.scores

    def get_state(self):
        head_x = self.heads[:, 0]
        head_y = self.heads[:, 1]
        
        point_l_x = head_x - 1
        point_r_x = head_x + 1
        point_u_y = head_y - 1
        point_d_y = head_y + 1
        
        def is_unsafe(x, y):
            return (x < 0) | (x >= self.grid_w) | (y < 0) | (y >= self.grid_h)

        dir_l = (self.directions[:, 0] == -1)
        dir_r = (self.directions[:, 0] == 1)
        dir_u = (self.directions[:, 1] == -1)
        dir_d = (self.directions[:, 1] == 1)
        
        danger_s = (dir_r & is_unsafe(point_r_x, head_y)) | \
                   (dir_l & is_unsafe(point_l_x, head_y)) | \
                   (dir_u & is_unsafe(head_x, point_u_y)) | \
                   (dir_d & is_unsafe(head_x, point_d_y))

        danger_r = (dir_r & is_unsafe(head_x, point_d_y)) | \
                   (dir_l & is_unsafe(head_x, point_u_y)) | \
                   (dir_u & is_unsafe(point_r_x, head_y)) | \
                   (dir_d & is_unsafe(point_l_x, head_y))
                   
        danger_l = (dir_r & is_unsafe(head_x, point_u_y)) | \
                   (dir_l & is_unsafe(head_x, point_d_y)) | \
                   (dir_u & is_unsafe(point_l_x, head_y)) | \
                   (dir_d & is_unsafe(point_r_x, head_y))
                   
        food_l = self.foods[:, 0] < head_x
        food_r = self.foods[:, 0] > head_x
        food_u = self.foods[:, 1] < head_y
        food_d = self.foods[:, 1] > head_y
        
        state = torch.stack([
            danger_s.float(), danger_r.float(), danger_l.float(),
            dir_l.float(), dir_r.float(), dir_u.float(), dir_d.float(),
            food_l.float(), food_r.float(), food_u.float(), food_d.float()
        ], dim=1)
        
        return state