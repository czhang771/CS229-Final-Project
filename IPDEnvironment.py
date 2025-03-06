import numpy as np
import random
import torch
from typing import List, Tuple, Dict

COOPERATE = 0
DEFECT = 1
ACTIONS = [COOPERATE, DEFECT]
PAYOFF_MATRIX = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1)
}

class IPDEnvironment:
    def __init__(self, payoff_matrix, num_rounds, k):
        self.payoff_matrix = payoff_matrix
        self.num_rounds = num_rounds
        self.payoff1 = 0
        self.payoff2 = 0
        self.k = k
        self.current_step = 0
        self.history = []
        self.reset()

    # Resets the environment state for a new game
    def reset(self, actor = 1):
        self.history = []
        self.current_step = 0
        return self.get_state(actor = actor)
    
    # Returns the most recent state (previous moves) for agents
    def get_state(self, actor = 1):
        if len(self.history) < self.k:
            # pad on left with 2s
            padded_history = [(2, 2)] * (self.k - len(self.history)) + self.history
            state = torch.tensor([item[:2] for item in padded_history])
        else:
            state = self.history[-self.k:]
            state = torch.tensor([item[:2] for item in state])
        
        if actor == 1:
            # recorded state should be THEIR ACTION, MY ACTION
            state = torch.flip(state, dims = [1])
        
        return state
    
    # Executes next round and updates state
    def step(self, action1, action2):
        # game has ended
        if self.current_step >= self.num_rounds:
            return None
        
        reward1, reward2 = self.payoff_matrix[(action1, action2)]
        self.payoff1 += reward1
        self.payoff2 += reward2
        
        self.history.append((action1, action2, reward1, reward2))
        self.current_step += 1
        # returns next state
        return self.get_state(), reward1, reward2

    def print_game_sequence(self):
        """ Prints the entire game history for debugging. """
        print("Game History:")
        for step, (a1, a2, r1, r2) in enumerate(self.history):
            print(f"Round {step+1}: P1 -> {a1}, P2 -> {a2} | Rewards: {r1}, {r2}")