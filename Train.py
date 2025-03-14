import torch
import numpy as np
from IPDEnvironment import IPDEnvironment
from Model import LogReg, MLP, LSTM, init_weights
from Learner import PolicyGradientLearner, ActorCriticLearner
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

    def __init__(self, env: IPDEnvironment, learner: PolicyGradientLearner, opponent: Union[Strategy, list[Strategy]], 
                 k: int = 2, gamma: float = 0.99, min_epsilon = 0.1, random_threshold = 0.8):
        self.env = env
        self.min_epsilon = min_epsilon
        self.learner = learner
        self.opponent = opponent
        self.n_opponents = len(opponent) if isinstance(opponent, list) else 1
        self.k = k
        self.score_history = []
        self.loss_history = []
        self.gamma = gamma
        self.random_threshold = random_threshold

    def rollout(self, game_length: int, num_games: int, epsilon_t: float = 1.0):
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
                action1 = self.learner.act(state, epsilon = epsilon_t, random_threshold = self.random_threshold)
                #action1 = random.choice([COOPERATE, DEFECT])
                action2 = opponent.act(state)
                next_state, reward1, reward2 = self.env.step(action1, action2)
            
            #self.env.print_game_sequence()
            
            # just to be explicitly clear about order
            if AGENT == 1: 
                trajectories.append(Trajectory(self.env.history, self.k, self.env.payoff1, self.env.payoff2))
            else: trajectories.append(Trajectory(self.env.history, self.k, self.env.payoff2, self.env.payoff1))

            #print(self.env.history)
        return trajectories

    def evaluate(self, game_length: int, num_games: int, eval_opponent: Strategy = None):
        """Evaluate against (potentially) different opponent"""
        if eval_opponent:
            original_opponent = self.opponent
            self.opponent = eval_opponent
        
        trajectories = self.rollout(game_length, num_games)
        
        # reset if necessary
        if eval_opponent: self.opponent = original_opponent
        
        scores = sum([t.my_payoff for t in trajectories])   
        #print(f"Score: {score}")
        return scores
    
    def train_MC(self, epochs: int, game_length: int, num_games: int, entropy_coef: float = 0.0):
        """Basic implementation of a train loop"""
        i = 0
        window_size = 3
        reward_threshold = 0.001 * PAYOFF_MATRIX[COOPERATE, COOPERATE][0] * game_length
        for i in range(epochs):
            if i > 2 * window_size and (np.mean(self.score_history[-window_size:]) - 
                      np.mean(self.score_history[-2*window_size:-window_size])) < reward_threshold:
                break
            # do rollouts
            epsilon_t = min(self.min_epsilon, 1.0 - i / epochs)
            trajectories = self.rollout(game_length, num_games, epsilon_t = epsilon_t)
            
            # update step
            self.learner.optimizer.zero_grad()
            loss = self.learner.loss(trajectories, gamma = self.gamma, entropy_coef = entropy_coef)
            loss.backward()
            self.learner.optimizer.step()

            # logging
            self.score_history.append(sum([t.my_payoff for t in trajectories]) / len(trajectories)) 
            self.loss_history.append(loss.item())
            if i % 1 == 0:
                print(f"Epoch {i}, score: {self.score_history[-1]}, loss: {self.loss_history[-1]}")
        
        return i

    def train_AC(self, epochs: int, game_length: int, num_games: int, batch_size: int = 5, entropy_coef: float = 0.0):
        """Train using actor-critic"""
        window_size = 3

        # REWARD PLATEAU THRESHOLD
        reward_threshold = 0.001 * PAYOFF_MATRIX[COOPERATE, COOPERATE][0] * game_length
        for i in range(epochs):
            if i > 2 * window_size and (np.mean(self.score_history[-window_size:]) - 
                      np.mean(self.score_history[-2*window_size:-window_size])) < reward_threshold:
                break
            
            epsilon_t = min(0.1, 1.0 - i / epochs)
            if batch_size > game_length:
                raise ValueError("Batch size must be less than game length")
            
            self.train_AC_batch(game_length, num_games, batch_size, epsilon_t, entropy_coef)
            self.learner.actor_optimizer.scheduler_step()
            self.learner.critic_optimizer.scheduler_step()

            if i % 1 == 0:
                print(f"Epoch {i}, score: {self.score_history[-1]}, actor loss: {self.loss_history[-1][0]}, critic loss: {self.loss_history[-1][1]}")
        
        return i

    def train_AC_batch(self, game_length: int, num_games: int, batch_size: int, epsilon_t = 1.0, entropy_coef: float = 0.0):
        # (inefficient) parallel environments
        envs = [self.env.copy() for _ in range(num_games)]
        states = torch.zeros(num_games, self.k, 2)

        # initialize first values
        for i in range(num_games):
            states[i, :, :] = envs[i].get_state(actor = AGENT)
        
        # initialize losses
        for _ in range(game_length // batch_size):
            all_states = []
            all_actions = []
            all_rewards = []
            all_next_states = []
            all_next_actions = []

            # loop through batch of timesteps
            for j in range(batch_size):
                actions = torch.zeros(num_games, dtype = torch.long)
                rewards = torch.zeros(num_games)
                next_states = torch.zeros(num_games, self.k, 2)
                next_actions = torch.zeros(num_games, dtype = torch.long)
                
                for i in range(num_games):
                    # sampling loop
                    actions[i] = self.learner.act(states[i], epsilon = epsilon_t, random_threshold = self.random_threshold)
                    #action2 = self.opponent.act(states[i])
                    action2 = random.choice(self.opponent).act(states[i])
                    next_state, reward1, reward2 = envs[i].step(int(actions[i]), int(action2))
                    rewards[i] = reward1
                    next_states[i, :, :] = next_state
                    next_actions[i] = self.learner.act(next_states[i], epsilon = 0.0) # greedy for target value
                
                # update buffer
                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_next_states.append(next_states)
                all_next_actions.append(next_actions)

                # update states
                states = next_states

            # stack data
            batch_states = torch.cat(all_states, dim = 0)
            batch_actions = torch.cat(all_actions, dim = 0)
            batch_rewards = torch.cat(all_rewards, dim = 0)
            batch_next_states = torch.cat(all_next_states, dim = 0)
            batch_next_actions = torch.cat(all_next_actions, dim = 0)
            
            # normalize rewards (added 3.10)
            normalized_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-6)

            # update actor model
            self.learner.actor_optimizer.zero_grad()
            actor_loss = self.learner.actor_loss(batch_states, batch_actions, entropy_coef)
            torch.nn.utils.clip_grad_norm_(self.learner.actor_model.parameters(), max_norm = 1.0)
            actor_loss.backward()
            self.learner.actor_optimizer.step()

            # update critic model
            self.learner.critic_optimizer.zero_grad()
            critic_loss = self.learner.critic_loss(batch_states, batch_actions, normalized_rewards, batch_next_states, batch_next_actions, gamma = self.gamma)
            torch.nn.utils.clip_grad_norm_(self.learner.critic_model.parameters(), max_norm = 1.0)
            critic_loss.backward()
            self.learner.critic_optimizer.step()

            #envs[0].print_game_sequence()

        #envs[0].print_game_sequence()
        # logging
        trajectories = []
        for i in range(num_games):
            trajectories.append(Trajectory(envs[i].history, self.k, envs[i].payoff1, envs[i].payoff2))
        self.score_history.append(sum([t.my_payoff for t in trajectories]) / len(trajectories))
        self.loss_history.append((actor_loss.item(), critic_loss.item()))

        return critic_loss.item(), actor_loss.item(), trajectories

    def load(self, path: str):
        self.learner.model.load_state_dict(torch.load(path))

    def plot_score(self, path: str):
        # plot score over time
        plt.plot(self.score_history)
        plt.savefig(path)
    

if __name__ == "__main__":
    # set random seed

    # non terminal reward basically doesn't work!
    k = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = IPDEnvironment(payoff_matrix = PAYOFF_MATRIX, num_rounds=1000, k = k)
    #opponent = Strong() 
    opponent = Random()
    
    # POLICY GRADIENTS EXAMPLES
    learner = PolicyGradientLearner(LogReg(d_input = STATE_DIM * k, d_output = NUM_ACTIONS), device, "adam", terminal = False, param_dict = {"lr": 0.05})
    # learner = PolicyGradientLearner(MLP(d_input = STATE_DIM * k, d_output = NUM_ACTIONS, d_hidden = [4 * k, 4 * k]), device, "adamw", terminal = False, param_dict = {"lr": 0.01})
    # learner = PolicyGradientLearner(LSTM(d_input = STATE_DIM, d_output = NUM_ACTIONS, d_hidden = [8 * STATE_DIM, 4 * STATE_DIM]), device, "adamw", terminal = True, param_dict = {"lr": 0.01})
    trainer = Trainer(env, learner, opponent, k = k, gamma = 0.99, random_threshold = 0.8, min_epsilon = 0.5)
    trainer.train_MC(epochs = 50, num_games = 10, game_length = 20, entropy_coef = 0.1)
    print(f"✅ Final Score for Agent: {env.payoff1}")
    print(f"✅ Final Score for Opponent: {env.payoff2}")

    # trainer.evaluate(game_length = 20, num_games = 10, eval_opponent = Du())

    # ACTOR-CRITIC EXAMPLES
    #actor = LogReg(d_input = STATE_DIM * k, d_output = NUM_ACTIONS)
    #critic = LogReg(d_input = STATE_DIM * k, d_output = NUM_ACTIONS)
    # actor = MLP(d_input = STATE_DIM * k, d_output = NUM_ACTIONS, d_hidden = [4 * k, 4 * k])
    # critic = MLP(d_input = STATE_DIM * k, d_output = NUM_ACTIONS, d_hidden = [8 * k, 4 * k, 4 * k])
    #actor = LSTM(d_input = STATE_DIM, d_output = NUM_ACTIONS, d_hidden = [4 * STATE_DIM, 8 * STATE_DIM, 4 * STATE_DIM])
    #critic = LSTM(d_input = STATE_DIM, d_output = NUM_ACTIONS, d_hidden = [4 * STATE_DIM, 8 * STATE_DIM, 4 * STATE_DIM])

    # IT REALLY WORKS A LOT BETTER IF THE THE CRITIC IS AN LSTM
    # THE HYPERPARAMETER TUNING IS REALLY ANNOYING

    # TODO:
    # make LSTM only use actual history (not padding)
    # figure out a better way than padding?

    # learner = ActorCriticLearner(actor, critic, device, 
    #                              actor_optimizer = "adamw", 
    #                              critic_optimizer = "adamw", 
    #                              terminal = False, 
    #                              param_dict = {"actor": {"lr": 0.005, "scheduler_type":"exponential", "scheduler_params": {"gamma": 0.999}},
    #                                             "critic": {"lr": 0.001, "scheduler_type":"exponential", "scheduler_params": {"gamma": 0.999}} })
    
    # trainer = Trainer(env, learner, opponent, k = k, gamma = 0.99, random_threshold = 0.5, min_epsilon = 0.1)
    
    # trainer.train_AC(epochs = 50, game_length = 20, num_games = 10, batch_size = 10, entropy_coef = 0.1)
    # print(trainer.evaluate(game_length = 20, num_games = 1, eval_opponent = Du()))
    
    # fig, ax = plt.subplots(3, 1)
    # ax[0].set_title("Actor-Critic Training")
    # # y axis labels
    # ax[0].set_ylabel("avg. cum. reward")
    # ax[1].set_ylabel("actor loss")
    # ax[2].set_ylabel("critic loss")
    # ax[0].plot(trainer.score_history, c = 'k')
    # ax[1].plot([x[0] for x in trainer.loss_history], c = 'k')
    # ax[2].plot([x[1] for x in trainer.loss_history], c = 'k')
    # plt.tight_layout()
    # #plt.savefig("ac_curve.pdf")
    # plt.show()