import torch
import numpy as np
from IPDEnvironment import IPDEnvironment
from Model import LogReg, MLP, LSTM
from Learner import PolicyGradientLearner, ActorCriticLearner, DQNLearner, PPOLearner
from Strategy import *
from Trajectory import Trajectory
import matplotlib.pyplot as plt

AGENT = 1

class Trainer:
    """Base class for all experiments."""

    def __init__(self, env: IPDEnvironment, learner: PolicyGradientLearner, opponent: Strategy, history_length: int = 2):
        self.env = env
        self.learner = learner
        self.opponent = opponent
        self.history_length = history_length
        self.score_history = []

    def rollout(self, game_length: int, num_games: int):
        """Play num_games of length game_length against opponent, return trajectories"""
        trajectories = []
        for _ in range(num_games):
            self.env.reset()
            for _ in range(game_length):
                state = self.env.get_state(k = self.history_length, actor = AGENT)
                action1 = self.learner.act(state)
                action2 = self.opponent.act(state)
                self.env.step(action1, action2)
            
            # just to be explicitly clear about order
            if AGENT == 1: trajectories.append(Trajectory(self.env.history, self.history_length, self.env.payoff1, self.env.payoff2))
            else: trajectories.append(Trajectory(self.env.history, self.history_length, self.env.payoff2, self.env.payoff1))
        
        return trajectories

    def evaluate(self, game_length: int, num_games: int, eval_opponent: Strategy = None):
        if eval_opponent:
            original_opponent = self.opponent
            self.opponent = eval_opponent
        
        trajectories = self.rollout(game_length, num_games)
        
        # reset if necessary
        if eval_opponent: self.opponent = original_opponent
        
        return sum([t.my_payoff for t in trajectories]) / num_games
    
    
    def train(self, epochs: int, game_length: int, num_games: int):
        """Basic implementation of a train loop"""
        for i in range(epochs):
            # do rollouts
            trajectories = self.rollout(game_length, num_games)
            # update step
            self.learner.optimizer.zero_grad()
            loss = self.learner.loss(trajectories, gamma = 0.99)
            loss.backward()
            self.learner.optimizer.step()

            self.score_history.append(sum([t.total_payoff for t in trajectories]) / len(trajectories)) 

            if i % 50 == 0:
                print(f"Epoch {i}, score: {self.score_history[-1]}")

    def save(self, path: str):
        torch.save(self.learner.model.state_dict(), path)

    def load(self, path: str):
        self.learner.model.load_state_dict(torch.load(path))

    def plot_score(self, path: str):
        # plot score over time
        plt.plot(self.score_history)
        plt.savefig(path)