import torch
import numpy as np
from abc import ABC, abstractmethod
from Trajectory import Trajectory
import Optimizer
from Model import Model

# states should be tensor of shape (n, 2)
# actions should be tensor of shape (n, 1)
# rewards should be tensor of shape (n, 1)
# but note that the policy will see *the last k steps in the trajectory* (padded) as input

class Learner(ABC):
    """Base class for all reinforcement learning algorithms (including optimizer + model)"""
    def __init__(self, model: Model, device: torch.device, optimizer_name: Optimizer, terminal: bool = True, param_dict = {}):
        self.model = model
        self.device = device
        self.terminal = terminal
        self.optimizer = Optimizer.create_optimizer(self.model, optimizer_name, param_dict)
    
    def act(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """Act on a state with epsilon-greedy policy; set 0 for greedy, 1 for completely random"""
        logits = self.model(state)
        if np.random.rand() < epsilon:
            # sample from distribution, NOT randomly
            action = torch.multinomial(torch.softmax(logits, dim = 1), num_samples = 1)
        else:
            # greedy
            action = torch.argmax(logits, dim = 1)
        
        return int(action)


class PolicyGradientLearner(Learner):
    """Basic REINFORCE policy gradient learner with cross-trajectory baseline"""
    def __init__(self, model: Model, device: torch.device, optimizer_name: str, terminal: bool = True, param_dict = {}):
        super().__init__(model, device, optimizer_name, terminal, param_dict)

    def loss(self, taus: list[Trajectory], gamma: float) -> torch.Tensor:
        # taus = list of trajectories
        losses = []
        all_Rt = []

        for trajectory in taus:
            R_t = trajectory.get_reward_sums(gamma = gamma, terminal = self.terminal)
            all_Rt.append(R_t)
        
        # print(all_Rt)
        
        # compute baseline as average of discounted rewards across all trajectories
        baseline = torch.cat(all_Rt).mean().detach()
        
        for i, trajectory in enumerate(taus):
            actions = trajectory.get_actions()
            states = trajectory.get_states()

            # compute advantage
            R_t = all_Rt[i]
            A_t = R_t - baseline

            # get logits, compute log probs, make sure properly batched
            logits = self.model(states, batched = True)
            log_probs = torch.log_softmax(logits, dim = 1)
            action_log_probs = torch.gather(log_probs, dim = 1, index = actions)
            
            # compute policy 'loss' (multiply by -1 to do EV maximization)
            policy_loss = -1 * torch.sum(action_log_probs * A_t)
            losses.append(policy_loss)
            # print(policy_loss)
        
        return torch.stack(losses).mean()


class ActorCriticLearner(Learner):
    def __init__(self, actor_model: Model, critic_model: Model, device: torch.device, actor_optimizer: str, critic_optimizer: str, terminal: bool = True, param_dict = {}):
        self.actor_model = actor_model
        self.critic_model = critic_model
        
        self.actor_optimizer = Optimizer.create_optimizer(self.actor_model, actor_optimizer, param_dict["actor"])
        self.critic_optimizer = Optimizer.create_optimizer(self.critic_model, critic_optimizer, param_dict["critic"])
        
        self.terminal = terminal
        self.device = device
    
    @property
    def model(self):
        return self.actor_model

    def actor_loss(self, states, actions):
        B = states.shape[0]
        actions = actions.view(B, 1)
        logits = self.actor_model(states, batched = True)
        log_probs = torch.log_softmax(logits, dim = 1)
        action_log_probs = torch.gather(log_probs, dim = 1, index = actions)

        # make sure Q values are not affecting gradient
        Q_values = self.critic_model(states, batched = True).detach()
        Q_values = torch.gather(Q_values, dim = 1, index = actions)
        
        actor_loss = -1 * torch.mean(action_log_probs * Q_values)
        return actor_loss

    def critic_loss(self, states, actions, rewards, next_states, next_actions, gamma):
        B = states.shape[0]
        actions = actions.view(B, 1)
        next_actions = next_actions.view(B, 1)
        Q_prev = self.critic_model(states, batched = True)
        Q_prev_values = torch.gather(Q_prev, dim = 1, index = actions)
        Q_next = self.critic_model(next_states, batched = True)
        Q_next_values = torch.gather(Q_next, dim = 1, index = next_actions)
        
        delta_t = rewards + gamma * Q_next_values - Q_prev_values
        
        # critic_loss = -1 * torch.mean(delta_t * Q_prev_values) also works but then must detach delta_t
        critic_loss = torch.mean(torch.pow(delta_t, 2))
        return critic_loss

class PPOLearner(Learner):
    def __init__(self, model, device):
        super().__init__(model, device)

    def act(self, state):
        pass