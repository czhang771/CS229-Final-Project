import numpy as np
import random 
from abc import ABC, abstractmethod

COOPERATE = 0
DEFECT = 1

class Strategy(ABC):
    """
    Abstract base class for all Iterated Prisoner's Dilemma strategies.
    """
    @abstractmethod
    def act(self, state):
        """
        Given the game state, return the action (COOPERATE or DEFECT).
        """
        pass

"""
Cooperates unconditionally.

"""
class Cu(Strategy):
    def act(self, state):
        return COOPERATE
    

"""
Defects unconditionally.
"""   
class Du(Strategy):
    def act(self, state):
        return DEFECT


"""
Cooperates with probability one-half.
"""
class Random(Strategy):
    def act(self, state):
        return random.choice([COOPERATE, DEFECT])

"""
Cooperates with fixed probably p
"""  
class Cp(Strategy):
    def __init__(self, p):
        self.p = p
    
    def act(self, state):
        return COOPERATE if random.random() < self.p else DEFECT

"""
Cooperates on the first round and imitates its opponent's previous move thereafter.
"""
class TFT(Strategy):
    def act(self, state):
        return COOPERATE if not state else state[1]

"""
Defects on the first round and imitates its opponent's previous move thereafter.
"""
class STFT(Strategy):
    def act(self, state):
        return DEFECT if not state else state[1]
 
"""
 Cooprates on the first round and after its opponent cooperates. Following a defection,it cooperates with probability 
 g(R,P, T, S) = min{1- (T-R)/(R-S), (R-P)/(T-P)}
"""   
class GTFT(Strategy):
    def __init__(self, R, P, T, S):
        self.R = R
        self.P = P
        self.T = T
        self.S = S

    def act(self, state):
        if not state:
            return COOPERATE
        prob = min(1- (self.T-self.T)/(self.R-self.S), (self.R-self.P)/(self.T-self.P))
        return COOPERATE if random.random() < prob else DEFECT

"""
TFT with two differences: 
(1) it increases the string of punishing defection responses with each additional defection by its opponent
(2) it apologizes for each string of defections by cooperating in the subsequent two rounds.
"""
class GrdTFT(Strategy):
    def __init__(self):
        self.punishment_length = 0  # How many rounds it should continue defecting
        self.apology_phase = 0  # Countdown for apology rounds

    def act(self, state):
        if state is None:
            return COOPERATE  

        last_opponent_move = state[1]

        # Apology phase: Cooperate for two rounds after prolonged punishment
        if self.apology_phase > 0:
            self.apology_phase -= 1
            return COOPERATE

        # If opponent defected, increase punishment length
        if last_opponent_move == DEFECT:
            self.punishment_length += 1
            return DEFECT  # Continue defecting for the punishment period

        # If opponent cooperated, check if we need to apologize
        if self.punishment_length > 0:
            self.punishment_length = 0  # Reset punishment counter
            self.apology_phase = 2  # Initiate apology phase
            return COOPERATE

        # Default TFT behavior (mirror opponent)
        return last_opponent_move

""" Imitates opponent's last move with high (but less than one) probability."""
class ImpTFT(Strategy):
    def __init__(self, p):
        self.p = p
    
    def act(self, state):
        if not state:
            return COOPERATE
        last_opponent_move = state[1]

        if random.random() < self.p:
            return last_opponent_move
        else:
            return COOPERATE if last_opponent_move == DEFECT else DEFECT

"""
Cooperates unless defected against twice in a row.
"""       
class TFTT(Strategy):
    def __init__(self):
        self.num_defected = 0

    def act(self, state):
        if state and state[1] == DEFECT:
            self.num_defected = 1
            return DEFECT
        
        if self.num_defected == 0:
            return COOPERATE
        self.num_defected -= 1
        return DEFECT

        
"""
Defects twice after being defected against, otherwise cooperates.
"""      
class TTFT(Strategy):
    def __init__(self):
        self.num_to_defect = 0

    def act(self, state):
        if state and state[1] == DEFECT:
            self.num_to_defect += 1
            return DEFECT

        return COOPERATE

"""
Cooperates until its opponent has defected once, and then defects for the rest of the game.
"""    
class GRIM(Strategy):
    def __init__(self):
        self.defect = False

    def act(self, state):
        if state[1] == DEFECT:
            self.defect = True
        
        return DEFECT if self.defect else COOPERATE

"""
Cooperates if it and its opponent moved alike in previous move and defects if they moved differently.
"""
class WSLS(Strategy):
    def act(self, state):
        if not state:
            return COOPERATE
        
        return COOPERATE if state[0] == state[1] else DEFECT

"""
Strong (gold, highly adaptable) opponent, implementation to be figured out
"""
class Strong(Strategy):
    def act(self, state):
        if not state:
            return COOPERATE
        
        return COOPERATE if state[0] == state[1] else DEFECT