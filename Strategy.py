import numpy as np
import random 
from abc import ABC, abstractmethod
from BaseLM import BaseLM
import torch

COOPERATE = 0
DEFECT = 1

def is_first_round(state):
    if state is None:
        return True, None
    
    last = state[-1][0]
    if isinstance(last, torch.Tensor):
        last = int(last.item())
    if last == 2:
        return True, None
    
    return False, last

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

    def reset(self):
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
    def __init__(self, p=0.7):
        self.p = p
    
    def act(self, state):
        return COOPERATE if random.random() < self.p else DEFECT

"""
Cooperates on the first round and imitates its opponent's previous move thereafter.
"""
class TFT(Strategy):
    # states are stored [agent action, strategy action]
    def act(self, state):
        is_first, move = is_first_round(state)
        if is_first:
            return COOPERATE
        else:
            return int(move)

"""
Defects on the first round and imitates its opponent's previous move thereafter.
"""
class STFT(Strategy):
    def act(self, state):
        is_first, move = is_first_round(state)
        if is_first:
            return DEFECT
        else:
            return int(move)
 
"""
 Cooprates on the first round and after its opponent cooperates. Following a defection,it cooperates with probability 
 g(R,P, T, S) = min{1- (T-R)/(R-S), (R-P)/(T-P)}
"""   
class GTFT(Strategy):
    def __init__(self, R=3, P=1, T=5, S=0):
        self.R = R
        self.P = P
        self.T = T
        self.S = S

    def act(self, state):
        is_first, move = is_first_round(state)
        if is_first:
            return COOPERATE
        
        if move == DEFECT:
            prob = min(1- (self.T-self.R)/(self.R-self.S), (self.R-self.P)/(self.T-self.P))
            return COOPERATE if random.random() < prob else DEFECT
        else:
            return COOPERATE

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
        is_first, move = is_first_round(state)
        if is_first:
            return COOPERATE  

        # Apology phase: Cooperate for two rounds after prolonged punishment
        if self.apology_phase > 0:
            self.apology_phase -= 1
            return COOPERATE

        # If opponent defected, increase punishment length
        if move == DEFECT:
            self.punishment_length += 1
            return DEFECT  # Continue defecting for the punishment period

        # If opponent cooperated, check if we need to apologize
        if self.punishment_length > 0:
            self.punishment_length = 0  # Reset punishment counter
            self.apology_phase = 2  # Initiate apology phase
            return COOPERATE

        # Default TFT behavior (mirror opponent)
        return move

    def reset(self):
        self.punishment_length = 0
        self.apology_phase = 0

""" Imitates opponent's last move with high (but less than one) probability."""
class ImpTFT(Strategy):
    def __init__(self, p=0.85):
        self.p = p
    
    def act(self, state):
        is_first, move = is_first_round(state)
        if is_first:
            return COOPERATE

        if random.random() < self.p:
            return move
        else:
            return COOPERATE if move == DEFECT else DEFECT

"""
Cooperates unless defected against twice in a row.
"""       
class TFTT(Strategy):
    def __init__(self):
        self.num_defected = 0

    def act(self, state):
        is_first, move = is_first_round(state)
        if is_first:
            return COOPERATE

        if move == DEFECT:
            self.num_defected += 1
            if self.num_defected > 2:
                self.num_defected = 0 # is this correct?
                return DEFECT
            else:
                return COOPERATE
        else:
            self.num_defected = 0
            return COOPERATE
        
    def reset(self):
        self.num_defected = 0

        
"""
Defects twice after being defected against, otherwise cooperates.
"""      
class TTFT(Strategy):
    def __init__(self):
        self.num_to_defect = 0

    def act(self, state):
        is_first, opp_move = is_first_round(state)
        if is_first:
            return COOPERATE
        
        # queue up defections
        if opp_move == DEFECT:
            self.num_to_defect = 2
        
        # if we have defections to make, do so and decrement counter
        if self.num_to_defect > 0:
            self.num_to_defect -= 1
            return DEFECT
        else:
            return COOPERATE
    
    def reset(self):
        self.num_to_defect = 0

"""
Cooperates until its opponent has defected once, and then defects for the rest of the game.
"""    
class GRIM(Strategy):
    def __init__(self):
        self.defect = False

    def act(self, state):
        is_first, opp_move = is_first_round(state)
        if is_first:
            return COOPERATE
        
        if opp_move == DEFECT:
            self.defect = True
        
        return DEFECT if self.defect else COOPERATE
    
    def reset(self):
        self.defect = False

"""
Cooperates if it and its opponent moved alike in previous move and defects if they moved differently.
"""
class WSLS(Strategy):
    def act(self, state):
        is_first, move1 = is_first_round(state)
        if is_first:
            return COOPERATE
        
        move2 = state[-1][1]
        
        return COOPERATE if move1 == move2 else DEFECT

"""
Strong (gold, highly adaptable) opponent, implementation to be figured out
"""
class Strong(Strategy):
    def __init__(self, model_config_path="LLMconfig.yaml"):
        self.model = BaseLM.from_config(model_config_path)
    def act(self, state):
        key = {COOPERATE: "COOPERATE", DEFECT: "DEFECT", 2:"UNKNOWN"}
        prompt = "Given the last {} rounds of an Iterated Prisoner's Dilemma in the format (opponent move, your move):\n".format(len(state))
        for i, (my_action, opp_action) in enumerate(state, 1):
            prompt += f"{i}. ({key[int(my_action.item())]}, {key[int(opp_action.item())]})\n"
        prompt += "\nShould you cooperate (0) or defect (1)? Remember, this is an iterated game, so always defecting is not necessarily the best strategy. Respond with only '0' or '1'."
        #print(prompt)
        response = self.model.generate_text(prompt).strip()
        #print(response)
        if response == "0":
            return COOPERATE
        elif response == "1":
            return DEFECT
        else:
            print(f"Warning: Unexpected LLM response: {response}. Defaulting to COOPERATE.")
            return COOPERATE