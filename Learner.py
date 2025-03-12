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
    
    def act(self, state: torch.Tensor, epsilon: float = 0.0, random_threshold: float = 0.8) -> int:
        """Act on a state with epsilon-greedy policy; set 0 for greedy, 1 for completely random"""
        logits = self.model(state)

        # Detect and replace NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: NaN or Inf detected in logits. Clamping values.")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e3, neginf=-1e3)  # Replace NaN with 0, Inf with large finite values

        # Clamp logits to prevent extreme values that could break softmax
        logits = torch.clamp(logits, min=-20, max=20)

        # Apply softmax and check for valid probabilities
        probs = torch.softmax(logits, dim=1)

        # Ensure probabilities are valid before sampling
        if torch.isnan(probs).any() or (probs < 0).any() or torch.isinf(probs).any() or torch.sum(probs) <= 0:
            print(f"Warning: Invalid probabilities detected, replacing with uniform distribution.")
            probs = torch.full_like(probs, 1.0 / probs.shape[1])  # Replace with uniform probabilities

        if np.random.rand() < epsilon:
            if np.random.rand() < random_threshold:
                # Ensure `probs` sum is valid before calling `multinomial`
                if torch.sum(probs) <= 0 or torch.any(probs < 0):
                    print(f"Warning: Invalid probability distribution, replacing with uniform distribution.")
                    probs = torch.full_like(probs, 1.0 / probs.shape[1])  # Reset to uniform probabilities

                action = torch.multinomial(probs, num_samples=1)
            else:
                action = torch.randint(0, probs.shape[1], (1,))
        else:
            # Greedy selection
            action = torch.argmax(probs, dim=1)

        return int(action)


class PolicyGradientLearner(Learner):
    """Basic REINFORCE policy gradient learner with cross-trajectory baseline"""
    def __init__(self, model: Model, device: torch.device, optimizer_name: str, terminal: bool = True, param_dict = {}):
        super().__init__(model, device, optimizer_name, terminal, param_dict)

    def loss(self, taus: list[Trajectory], gamma: float, entropy_coef: float = 0.0) -> torch.Tensor:
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
            probs = torch.softmax(logits, dim = 1)
            action_log_probs = torch.gather(log_probs, dim = 1, index = actions)

            gamma_prods = torch.tensor([gamma**i for i in range(len(actions))])
            # gamma_prods = torch.ones_like(actions)
            # hadamard product
            A_t = A_t * gamma_prods
            
            # compute policy 'loss' (multiply by -1 to do EV maximization)
            policy_loss = -1 * torch.sum(action_log_probs * A_t)
            entropy = -1 * torch.sum(probs * log_probs, dim = 1)
            # we want to maximize entropy
            policy_loss = policy_loss - entropy_coef * entropy.mean()
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

    def actor_loss(self, states, actions, entropy_coef = 0.2):
        # states: shape B x (2k)
        B = states.shape[0]
        actions = actions.view(B, 1)
        logits = self.actor_model(states, batched = True)
        log_probs = torch.log_softmax(logits, dim = 1)
        probs = torch.softmax(logits, dim = 1)
        action_log_probs = torch.gather(log_probs, dim = 1, index = actions)

        # make sure Q values are not affecting gradient
        Q_values = self.critic_model(states, batched = True).detach()
        Q_values = torch.gather(Q_values, dim = 1, index = actions)
        
        actor_loss = -1 * torch.mean(action_log_probs * Q_values)   
        entropy = -1 * torch.sum(probs * log_probs, dim = 1)
        # we want to maximize entropy
        return actor_loss - entropy_coef * entropy.mean()

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