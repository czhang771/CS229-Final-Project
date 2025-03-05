import torch
import numpy as np
from abc import ABC, abstractmethod
from Trajectory import Trajectory

# states should be tensor of shape (n, 2)
# actions should be tensor of shape (n, 1)
# rewards should be tensor of shape (n, 1)
# but note that the policy will see *the last k steps in the trajectory* (padded) as input

class Learner(ABC):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.terminal = True
    
    def act(self, state, epsilon = 0.0):
        logits = self.model(state)
        if np.random.rand() < epsilon:
            # sample from distribution, NOT randomly
            action = torch.multinomial(torch.softmax(logits, dim = 1), num_samples = 1)
        else:
            # greedy
            action = torch.argmax(logits, dim = 1)
        
        return action

    @abstractmethod
    def loss(self):
        pass


class PolicyGradientLearner(Learner):
    def __init__(self, model, device):
        super().__init__(model, device)

    def loss(self, taus: list[Trajectory], gamma: float):
        # taus = list of trajectories
        R_t = [taus[i].get_reward_sums(gamma = gamma, terminal = self.terminal) for i in range(0, len(taus))]
        R_t = torch.stack(R_t)
        
        # calculate advantage as the average across all states (boring)
        A_t = R_t - torch.mean(R_t, dim = 0)

        # compute log probs
        logits = torch.stack(taus.get_states())
        log_probs = torch.log(torch.softmax(logits, dim = 1))
        action_log_probs = torch.gather(log_probs, dim = 1, index = taus.get_actions())
        
        policy_loss = -1 * torch.sum(action_log_probs * A_t)
        return policy_loss


class ActorCriticLearner(Learner):
    def __init__(self, model, device):
        super().__init__(model, device)

    def act(self, state):
        pass


class DQNLearner(Learner):
    def __init__(self, model, device):
        super().__init__(model, device)

    def act(self, state):
        pass


class PPOLearner(Learner):
    def __init__(self, model, device):
        super().__init__(model, device)

    def act(self, state):
        pass


class GRPOLearner(Learner):
    def __init__(self, model, device):
        super().__init__(model, device)

    def act(self, state):
        pass
