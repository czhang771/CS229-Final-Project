import torch
import numpy as np
from IPDEnvironment import IPDEnvironment
from Model import LogReg, MLP, LSTM
from Learner import PolicyGradientLearner, ActorCriticLearner, PPOLearner
from Strategy import *
from Trajectory import Trajectory
import matplotlib.pyplot as plt
from typing import Union

AGENT = 1
PAYOFF_MATRIX = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1)
}

class Trainer:
    """Base class for all experiments."""

    def __init__(self, env: IPDEnvironment, learner: PolicyGradientLearner, opponent: Union[Strategy, list[Strategy]], k: int = 2):
        self.env = env
        self.learner = learner
        self.opponent = opponent
        self.n_opponents = len(opponent) if isinstance(opponent, list) else 1
        self.k = k
        self.score_history = []

    def rollout(self, game_length: int, num_games: int):
        """Play num_games of length game_length against opponent (randomly selected), return trajectories"""
        
        trajectories = []

        # outer game loop; play many games per epoch
        for _ in range(num_games):
            self.env.reset()
            if self.n_opponents == 1:
                opponent = self.opponent
            else:
                opponent = random.choice(self.opponent)
            
            # inner game loop; agents do not know game length
            for _ in range(game_length):
                state = self.env.get_state(actor = AGENT)
                action1 = self.learner.act(state)
                action2 = opponent.act(state)
                next_state, reward1, reward2 = self.env.step(action1, action2)
            
            # just to be explicitly clear about order
            if AGENT == 1: trajectories.append(Trajectory(self.env.history, self.k, self.env.payoff1, self.env.payoff2))
            else: trajectories.append(Trajectory(self.env.history, self.k, self.env.payoff2, self.env.payoff1))
        
        return trajectories

    def evaluate(self, game_length: int, num_games: int, eval_opponent: Strategy = None):
        """Evaluate against (potentially) different opponent"""
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

            # logging
            self.score_history.append(sum([t.my_payoff for t in trajectories]) / len(trajectories)) 

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
    

if __name__ == "__main__":
    k = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = IPDEnvironment(payoff_matrix = PAYOFF_MATRIX, num_rounds = 1000, k = k)
    learner = PolicyGradientLearner(LogReg(d_input = 2 * k, d_output = 2), device)
    opponent = Random()
    trainer = Trainer(env, learner, opponent)
    trainer.train(1000, 1000, 10)