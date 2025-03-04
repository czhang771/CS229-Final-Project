import torch
import numpy as np
from abc import ABC, abstractmethod

class Learner(ABC):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass



class PolicyGradientLearner(Learner):
    def __init__(self, model, device):
        super().__init__(model, device)

    def act(self, state):
        pass



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