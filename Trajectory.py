import torch
import numpy as np
from collections import defaultdict

class Trajectory:
    def __init__(self, history, k):
        # initialize trajectory from history of tuples (opponent_action, action, my_reward, opponent_reward)
        self.history = history
        self.k = k
        
        # make into tensors
        if history:
            self.opponent_actions = torch.tensor([h[0] for h in history])
            self.actions = torch.tensor([h[1] for h in history])
            self.rewards = torch.tensor([h[2] for h in history]).float()
            self.opponent_rewards = torch.tensor([h[3] for h in history]).float()
        else:
            self.opponent_actions = torch.tensor([])
            self.actions = torch.tensor([])
            self.rewards = torch.tensor([])
            self.opponent_rewards = torch.tensor([])
        
        # game length
        self.length = len(history)
    
    def get_states(self) -> torch.Tensor:
        if self.length == 0:
            return torch.empty((0, self.k, 2))
        
        # use 2 padding (one-hot for states, padding dimension)
        padded_states = torch.full((self.length, self.k, 2), 2)
        
        for i in range(self.length):
            for j in range(self.k):
                if i - j >= 0:
                    padded_states[i, j, 0] = self.opponent_actions[i - j]
                    padded_states[i, j, 1] = self.actions[i - j]
        
        # return n x k x 2 tensor of states
        return padded_states
    
    def get_actions(self) -> torch.Tensor:
        # return n x 1 tensor of actions taken by model
        if self.length == 0:
            return torch.empty((0, 1))
        
        return self.actions.view(-1, 1)
    
    def get_unique_states(self) -> dict:
        if self.length == 0:
            return {}
        
        # group unique states
        states = self.get_states()
        unique_states = defaultdict(list)
        
        for i in range(self.length):
            state_tuple = tuple(states[i].flatten().tolist())
            unique_states[state_tuple].append(i)
            
        return dict(unique_states)
    
    def get_reward_sums(self, gamma=0.99, terminal=False) -> torch.Tensor:
        if self.length == 0:
            return torch.empty((0, 1))
        
        if terminal:
            terminal_reward = self.rewards[-1]
            discounts = torch.tensor([gamma ** (self.length - i - 1) for i in range(self.length)])
            reward_sums = terminal_reward * discounts
        else: # dp work backwards
            reward_sums = torch.zeros(self.length)
            reward_sums[-1] = self.rewards[-1]
            for i in range(self.length - 2, -1, -1):
                reward_sums[i] = self.rewards[i] + gamma * reward_sums[i + 1]
        
        return reward_sums.view(-1, 1)