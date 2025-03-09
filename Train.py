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
NUM_ACTIONS = 2
STATE_DIM = 2
PAYOFF_MATRIX = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1)
}

class Trainer:
    """Base class for all experiments."""

    def __init__(self, env: IPDEnvironment, learner: PolicyGradientLearner, opponent: Union[Strategy, list[Strategy]], k: int = 2, gamma: float = 0.99):
        self.env = env

        self.learner = learner
        self.opponent = opponent
        self.n_opponents = len(opponent) if isinstance(opponent, list) else 1
        self.k = k
        self.score_history = []
        self.loss_history = []
        self.gamma = gamma

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
            opponent.reset()
            
            # inner game loop; agents do not know game length
            for _ in range(game_length):
                state = self.env.get_state(actor = AGENT)
                # act with full sampling
                # assume simultaneous actions
                action1 = self.learner.act(state, epsilon = 1.0)
                action2 = opponent.act(state)
                next_state, reward1, reward2 = self.env.step(action1, action2)
            
            # env.print_game_sequence()
            
            # just to be explicitly clear about order
            if AGENT == 1: 
                trajectories.append(Trajectory(self.env.history, self.k, self.env.payoff1, self.env.payoff2))
            else: trajectories.append(Trajectory(self.env.history, self.k, self.env.payoff2, self.env.payoff1))

            # print(self.env.history)
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
    
    
    def train_MC(self, epochs: int, game_length: int, num_games: int):
        """Basic implementation of a train loop"""
        
        for i in range(epochs):
            # do rollouts
            trajectories = self.rollout(game_length, num_games)
            
            # update step
            self.learner.optimizer.zero_grad()
            loss = self.learner.loss(trajectories, gamma = self.gamma)
            loss.backward()
            self.learner.optimizer.step()

            # logging
            self.score_history.append(sum([t.my_payoff for t in trajectories]) / len(trajectories)) 
            self.loss_history.append(loss.item())
            if i % 1 == 0:
                print(f"Epoch {i}, score: {self.score_history[-1]}, loss: {self.loss_history[-1]}")

    def train_AC(self, epochs: int, game_length: int, num_games: int):
        """Train using actor-critic"""
        for i in range(epochs):
            self.train_batch_step(game_length, num_games)

            if i % 1 == 0:
                print(f"Epoch {i}, score: {self.score_history[-1]}, actor loss: {self.loss_history[-1][0]}, critic loss: {self.loss_history[-1][1]}")

    def train_batch_step(self, game_length: int, num_games: int):
        # (inefficient) parallel environments
        envs = [self.env.copy() for _ in range(num_games)]

        states = torch.zeros(num_games, self.k, 2)
        actions = torch.zeros(num_games, dtype = torch.long)
        rewards = torch.zeros(num_games)
        next_states = torch.zeros(num_games, self.k, 2)
        next_actions = torch.zeros(num_games, dtype = torch.long)

        # initialize first values
        for i in range(num_games):
            states[i, :, :] = envs[i].get_state(actor = AGENT)
            actions[i] = self.learner.act(states[i], epsilon = 1.0)
        
        for _ in range(game_length):
            for i in range(num_games):
                # sampling loop
                action2 = self.opponent.act(states[i])
                next_state, reward1, reward2 = envs[i].step(int(actions[i]), int(action2))
                rewards[i] = reward1
                next_states[i, :, :] = next_state

                next_action = self.learner.act(next_states[i], epsilon = 1.0)
                next_actions[i] = next_action
            
            # compute update for actor
            self.learner.actor_optimizer.zero_grad()
            actor_loss = self.learner.actor_loss(states, actions)
            actor_loss.backward()
            self.learner.actor_optimizer.step()

            # compute update for critic
            self.learner.critic_optimizer.zero_grad()
            critic_loss = self.learner.critic_loss(states, actions, rewards, next_states, next_actions, gamma = self.gamma)
            critic_loss.backward()
            self.learner.critic_optimizer.step()
            
            # update states and actions
            states = next_states
            actions = next_actions
        
        # logging
        trajectories = []
        for i in range(num_games):
            trajectories.append(Trajectory(envs[i].history, self.k, envs[i].payoff1, envs[i].payoff2))
        self.score_history.append(sum([t.my_payoff for t in trajectories]) / len(trajectories))
        self.loss_history.append((actor_loss.item(), critic_loss.item()))

    def load(self, path: str):
        self.learner.model.load_state_dict(torch.load(path))

    def plot_score(self, path: str):
        # plot score over time
        plt.plot(self.score_history)
        plt.savefig(path)
    

if __name__ == "__main__":
    # non terminal reward basically doesn't work!
    k = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = IPDEnvironment(payoff_matrix = PAYOFF_MATRIX, num_rounds = 1000, k = k)
    opponent = TFT() 
    
    # POLICY GRADIENTS EXAMPLES
    # learner = PolicyGradientLearner(LogReg(d_input = STATE_DIM * k, d_output = NUM_ACTIONS), device, "adam", terminal = False, param_dict = {"lr": 0.05})
    # learner = PolicyGradientLearner(MLP(d_input = STATE_DIM * k, d_output = NUM_ACTIONS, d_hidden = [4 * k, 4 * k]), device, "adamw", terminal = False, param_dict = {"lr": 0.01})
    # learner = PolicyGradientLearner(LSTM(d_input = STATE_DIM, d_output = NUM_ACTIONS, d_hidden = [8 * STATE_DIM, 4 * STATE_DIM]), device, "adamw", terminal = True, param_dict = {"lr": 0.01})
    # trainer = Trainer(env, learner, opponent, k = k)
    # trainer.train_MC(epochs = 40, num_games = 10, game_length = 20)

    # ACTOR-CRITIC EXAMPLES
    # actor = LogReg(d_input = STATE_DIM * k, d_output = NUM_ACTIONS)
    # critic = LogReg(d_input = STATE_DIM * k, d_output = NUM_ACTIONS)
    actor = MLP(d_input = STATE_DIM * k, d_output = NUM_ACTIONS, d_hidden = [4 * k, 4 * k])
    critic = MLP(d_input = STATE_DIM * k, d_output = NUM_ACTIONS, d_hidden = [4 * k, 4 * k])
    # actor = LSTM(d_input = STATE_DIM, d_output = NUM_ACTIONS, d_hidden = [4 * STATE_DIM, 8 * STATE_DIM, 4 * STATE_DIM])
    # critic = LSTM(d_input = STATE_DIM, d_output = NUM_ACTIONS, d_hidden = [4 * STATE_DIM, 8 * STATE_DIM, 4 * STATE_DIM])
    
    learner = ActorCriticLearner(actor, critic, device, actor_optimizer = "adamw", critic_optimizer = "adamw", terminal = False, param_dict = {"actor": {"lr": 0.001}, "critic": {"lr": 0.001} })
    trainer = Trainer(env, learner, opponent, k = k, gamma = 0.99)
    trainer.train_AC(epochs = 40, game_length = 20, num_games = 5)
    plt.plot(trainer.score_history)
    plt.show()