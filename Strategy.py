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
    def act(self, state, epsilon = 0.0, random_threshold = 0.8):
        return COOPERATE
    

"""
Defects unconditionally.
"""   
class Du(Strategy):
    def act(self, state, epsilon = 0.0, random_threshold = 0.8):
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
        prompt += "\nShould you cooperate (0) or defect (1)? You use the following strategy. First, remember this is an iterated game, so always defecting is not necessarily the best strategy. However, you should observe your opponent and exploit any overly cooperative patterns. You try to balance forgiveness and retaliation. You avoid *only* defecting. Respond with only '0' or '1'."
        # print(prompt)
        response = self.model.generate_text(prompt).strip()
        print(response)
        if response == "0":
            return COOPERATE
        elif response == "1":
            return DEFECT
        else:
            print(f"Warning: Unexpected LLM response: {response}. Defaulting to COOPERATE.")
            return COOPERATE
import random
from collections import defaultdict

class AdaptiveMemoryStrategy(Strategy):
    """
    A strong IPD strategy that combines elements from the top-performing strategies
    in the Axelrod tournaments, optimized to exploit cooperative opponents.
    
    Key features:
    1. Starts cooperatively but looks for exploitation opportunities
    2. Maintains a memory of opponent's behavior patterns
    3. Adapts to opponent's strategy over time
    4. Selectively forgives but prioritizes exploitation when possible
    5. Handles noise with a pattern recognition system
    """
    
    def __init__(self, memory_depth=10, noise_tolerance=0.05, exploit_threshold=0.8):
        # Configuration parameters
        self.memory_depth = memory_depth
        self.noise_tolerance = noise_tolerance
        self.exploit_threshold = exploit_threshold  # Threshold to detect exploitable opponents
        
        # State tracking
        self.history = []
        self.pattern_memory = defaultdict(lambda: {'cooperate': 0, 'defect': 0})
        self.cooperation_rate = 1.0
        self.exploitation_detected = False
        self.opponent_exploitable = False
        self.consecutive_defections = 0
        self.consecutive_cooperations = 0
        self.last_n_rounds = []
        self.exploitation_mode = False
        self.probe_cycle = 0
        
        # Exploitation parameters
        self.defection_counter = 0
        self.probe_frequency = 5  # How often to test opponent's reaction to defection
        
        # Initial state: cooperate on first move
        self.initial_action = COOPERATE

    def act(self, state):
        """
        Choose action based on game state.
        State format is a list of tuples (agent_action, opponent_action)
        """
        # Check if this is the first round
        is_first, opponent_last_move = is_first_round(state)
        if is_first:
            return self.initial_action
        
        # Update our history
        if len(state) > 0:
            if isinstance(state[-1][0], torch.Tensor):
                my_last_move = int(state[-1][1].item())
                opponent_last_move = int(state[-1][0].item())
            else:
                my_last_move = state[-1][1]
                opponent_last_move = state[-1][0]
                
            self.history.append((my_last_move, opponent_last_move))
            
            # Update cooperation rate
            if len(self.history) > 0:
                coop_count = sum(1 for _, move in self.history if move == COOPERATE)
                self.cooperation_rate = coop_count / len(self.history)
            
            # Update pattern memory with the last sequence
            if len(self.history) >= 2:
                # Track patterns of length 2
                pattern = self._get_pattern(2)
                if opponent_last_move == COOPERATE:
                    self.pattern_memory[pattern]['cooperate'] += 1
                else:
                    self.pattern_memory[pattern]['defect'] += 1
        
        # Update streak counters
        if opponent_last_move == DEFECT:
            self.consecutive_defections += 1
            self.consecutive_cooperations = 0
            # Exit exploitation mode if opponent retaliates
            if my_last_move == DEFECT and self.exploitation_mode:
                self.exploitation_mode = False
        else:
            self.consecutive_defections = 0
            self.consecutive_cooperations += 1
            
            # Check if opponent is potentially exploitable (forgives defections)
            if my_last_move == DEFECT and opponent_last_move == COOPERATE:
                self.defection_counter += 1
                if self.defection_counter >= 2:
                    self.opponent_exploitable = True
                    self.exploitation_mode = True
        
        # Keep track of last n rounds for pattern matching
        self.last_n_rounds.append(opponent_last_move)
        if len(self.last_n_rounds) > self.memory_depth:
            self.last_n_rounds.pop(0)
        
        # Detect if we're being exploited
        if self.consecutive_defections >= 2:
            self.exploitation_detected = True
            self.exploitation_mode = False  # Don't try to exploit if we're being exploited
        
        # Check if opponent is highly cooperative
        if len(self.history) >= 5:
            recent_coop_rate = sum(1 for _, move in self.history[-5:] if move == COOPERATE) / 5
            if recent_coop_rate >= self.exploit_threshold:
                # Opponent seems very cooperative, consider exploiting
                self.probe_cycle += 1
                if self.probe_cycle >= self.probe_frequency:
                    self.probe_cycle = 0
                    # Periodically test if opponent will forgive defection
                    if not self.exploitation_mode:
                        return DEFECT
        
        # If we've detected that the opponent is a forgiving type, enter exploitation mode
        if self.opponent_exploitable and self.consecutive_defections == 0:
            if random.random() < 0.8:  # 80% chance to exploit, 20% chance to cooperate to maintain trust
                self.exploitation_mode = True
            
        # Advanced detection of exploitation attempts by opponent
        if len(self.history) >= 3:
            # If opponent defected after our cooperation more than 70% of the time
            cooperate_then_defect = sum(1 for i in range(len(self.history)-1) 
                                       if self.history[i][0] == COOPERATE and self.history[i+1][1] == DEFECT)
            if cooperate_then_defect > 0:
                exploit_rate = cooperate_then_defect / sum(1 for m, _ in self.history if m == COOPERATE)
                if exploit_rate > 0.7:
                    self.exploitation_detected = True
                    self.exploitation_mode = False
        
        # Decision logic
        action = self._decide_action(opponent_last_move)
        
        return action

    def _decide_action(self, opponent_last_move):
        """Core decision logic based on observed patterns and current state"""
        # If we're in exploitation mode and opponent seems forgiving, defect
        if self.exploitation_mode:
            # Occasionally cooperate to prevent opponent from switching to permanent defection
            if self.consecutive_defections >= 3 or random.random() < 0.2:
                return COOPERATE
            return DEFECT
        
        # Always retaliate against multiple consecutive defections
        if self.consecutive_defections >= 2:
            return DEFECT
        
        # If opponent is highly cooperative (80%+ cooperation rate), periodically defect
        if self.cooperation_rate > 0.8 and len(self.history) > 10:
            # Test exploitation with increasing frequency as opponent proves more cooperative
            exploit_chance = min(0.3, (self.cooperation_rate - 0.8) * 2)
            if random.random() < exploit_chance:
                return DEFECT
        
        # If opponent defected last round but has been mostly cooperative, forgive
        if opponent_last_move == DEFECT and self.cooperation_rate > 0.7:
            # 70% chance to forgive a single defection from a cooperative opponent
            return COOPERATE if random.random() < 0.7 else DEFECT
        
        # Use pattern prediction if we have enough history
        if len(self.history) >= 2:
            pattern = self._get_pattern(2)
            if pattern in self.pattern_memory:
                coop_count = self.pattern_memory[pattern]['cooperate']
                defect_count = self.pattern_memory[pattern]['defect']
                total = coop_count + defect_count
                
                if total >= 2:  # Only use pattern if we've seen it enough times
                    if defect_count / total > 0.7:  # Pattern predicts defection
                        return DEFECT
                    elif coop_count / total > 0.7:  # Pattern predicts cooperation
                        # If opponent is predicted to cooperate, occasionally exploit
                        if random.random() < 0.3:
                            return DEFECT
                        return COOPERATE
        
        # If opponent has cooperated many times in a row, occasionally defect
        if self.consecutive_cooperations >= 4:
            return DEFECT if random.random() < 0.4 else COOPERATE
        
        # Handle exploitation attempts
        if self.exploitation_detected:
            # When being exploited, use a more defensive approach
            if self.cooperation_rate < 0.3:  # If opponent rarely cooperates
                return DEFECT
            else:
                # Try to reset relationship but be cautious
                self.exploitation_detected = False
                return COOPERATE if random.random() < 0.5 else DEFECT
        
        # TFT-like behavior as default, with slight bias toward defection
        return opponent_last_move if random.random() < 0.9 else DEFECT

    def _get_pattern(self, length):
        """Extract the recent pattern of plays of specified length"""
        if len(self.history) < length:
            return None
        
        # Convert recent history to a pattern string
        pattern = tuple(move for _, move in self.history[-length:])
        return pattern

    def reset(self):
        """Reset the strategy state between games"""
        self.history = []
        self.pattern_memory = defaultdict(lambda: {'cooperate': 0, 'defect': 0})
        self.cooperation_rate = 1.0
        self.exploitation_detected = False
        self.opponent_exploitable = False
        self.consecutive_defections = 0
        self.consecutive_cooperations = 0
        self.last_n_rounds = []
        self.exploitation_mode = False
        self.probe_cycle = 0
        self.defection_counter = 0