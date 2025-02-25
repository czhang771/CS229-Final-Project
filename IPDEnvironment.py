import numpy as np
import random
from typing import List, Tuple, Dict

COOPERATE = 0
DEFECT = 1
ACTIONS = [COOPERATE, DEFECT]

class IPDEnvironment:
    def __init__(self, payoff_matrix, num_rounds):
        self.payoff_matrix = payoff_matrix
        self.num_rounds = num_rounds
        self.payoff1 = 0
        self.payoff2 = 0
        self.reset()

    # Resets the environment state for a new game
    def reset(self):
        self.history = []
        self.current_iteration = 0
        return self.get_state()
    
    # Returns the most recent state (previous moves) for agents
    def get_state(self):
        return self.history[-1] if self.history else None
    
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
        return self.get_state()

    def print_game_sequence(self):
        """ Prints the entire game history for debugging. """
        print("Game History:")
        for step, (a1, a2, r1, r2) in enumerate(self.history):
            print(f"Round {step+1}: P1 -> {a1}, P2 -> {a2} | Rewards: {r1}, {r2}")