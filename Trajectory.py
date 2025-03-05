import torch
import numpy as np
from collections import defaultdict

class Trajectory:
    """Store trajectory of a single game and useful utility functions"""
    def __init__(self, history, k, my_payoff, opponent_payoff):
        # initialize trajectory from history of tuples (opponent_action, action, my_reward, opponent_reward)
        self.history = history
        self.k = k
        self.my_payoff = my_payoff # total reward, summed across games
        self.opponent_payoff = opponent_payoff # total reward, summed across games

        # make into tensors
        if history:
            # history is list of tuples (opponent_action, action, my_reward, opponent_reward)
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
        """
        Returns a tensor of shape (length, k, 2) where each element is the k-windowed state.
        States are arranged chronologically from left to right.
        Right padding with 2s is used when there isn't enough history.
        
        Examples (assuming k=3):
        - At position 0: [0's action, 2, 2]  (only 1 history item, rest is padding)
        - At position 1: [0's action, 1's action, 2]  (2 history items, 1 padding)
        - At position 2: [0's action, 1's action, 2's action]  (3 history items, no padding)
        - At position 3: [1's action, 2's action, 3's action]  (3 history items, no padding)
        """
        if self.length == 0:
            return torch.empty((0, self.k, 2))
            
        # Initialize all values with padding (2)
        padded_states = torch.full((self.length, self.k, 2), 2)
        
        for i in range(self.length):
            # For each position i in the sequence
            
            # Determine how many valid history items we have (can't exceed k)
            valid_items = min(i + 1, self.k)
            
            for j in range(valid_items):
                # j goes from 0 to valid_items-1
                # We want to place the oldest history first (at position 0)
                # and the newest history last (at position valid_items-1)
                
                # Calculate the actual history index to use
                history_idx = i - (valid_items - 1) + j
                
                # Fill in the action data
                padded_states[i, j, 0] = self.opponent_actions[history_idx]
                padded_states[i, j, 1] = self.actions[history_idx]
        
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
            terminal_reward = self.my_payoff
            discounts = torch.tensor([gamma ** (self.length - i - 1) for i in range(self.length)])
            reward_sums = terminal_reward * discounts
        else: # dp work backwards
            reward_sums = torch.zeros(self.length)
            reward_sums[-1] = self.rewards[-1]
            for i in range(self.length - 2, -1, -1):
                reward_sums[i] = self.rewards[i] + gamma * reward_sums[i + 1]
        
        return reward_sums.view(-1, 1)