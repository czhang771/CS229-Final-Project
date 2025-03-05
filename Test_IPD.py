import pytest
import numpy as np
import torch

from IPDEnvironment import IPDEnvironment
from Strategy import Strategy, TitForTat, AlwaysCooperate, AlwaysDefect, RandomStrategy
from Model import LogReg, MLP, LSTM
from Learner import PolicyGradientLearner
from Trajectory import Trajectory
from Train import Trainer
from Optimizer import OptimizerFactory

# Configurable payoff matrix as a global variable
# (Row player, Column player) for each outcome
# Format: (R,R), (S,T), (T,S), (P,P) 
# Where R=Reward for cooperation, S=Sucker's payoff, T=Temptation to defect, P=Punishment for mutual defection
# Default values set to standard Prisoner's Dilemma matrix
PAYOFF_MATRIX = [
    [(3, 3), (0, 5)],  # Row player cooperates
    [(5, 0), (1, 1)]   # Row player defects
]


class TestIPDEnvironment:
    def test_reset(self):
        env = IPDEnvironment(payoffs=PAYOFF_MATRIX)
        state = env.reset()
        assert env.t == 0
        assert len(env.history) == 0
        assert state.shape == (1, 0)  # Empty history at start
    
    def test_step_first_move(self):
        env = IPDEnvironment(payoffs=PAYOFF_MATRIX)
        env.reset()
        next_state, reward, done, _ = env.step(0, 0)  # Both cooperate
        
        # Check history
        assert len(env.history) == 1
        assert env.history[0] == (0, 0)
        
        # Check reward - should be R,R for mutual cooperation
        assert reward == PAYOFF_MATRIX[0][0][0]
        
        # Check done
        assert not done
        
        # Check state shape (first player gets history with opponent move)
        assert next_state.shape == (1, 1)
    
    def test_payoff_matrix(self):
        env = IPDEnvironment(payoffs=PAYOFF_MATRIX)
        
        # Test all possible outcomes according to Prisoner's Dilemma payoff matrix
        # CC: Both cooperate (R,R)
        assert env.payoffs[0][0] == PAYOFF_MATRIX[0][0]
        
        # CD: Player cooperates, opponent defects (S,T)
        assert env.payoffs[0][1] == PAYOFF_MATRIX[0][1]
        
        # DC: Player defects, opponent cooperates (T,S)
        assert env.payoffs[1][0] == PAYOFF_MATRIX[1][0]
        
        # DD: Both defect (P,P)
        assert env.payoffs[1][1] == PAYOFF_MATRIX[1][1]
    
    def test_full_game(self):
        env = IPDEnvironment(payoffs=PAYOFF_MATRIX, max_steps=5)
        env.reset()
        
        # Play a full game
        rewards = []
        for _ in range(5):
            _, reward, done, _ = env.step(0, 0)  # Always cooperate
            rewards.append(reward)
            
            if done:
                break
        
        assert len(rewards) == 5
        mutual_coop_reward = PAYOFF_MATRIX[0][0][0]
        assert all(r == mutual_coop_reward for r in rewards)  # All mutual cooperation
        assert len(env.history) == 5


class TestStrategies:
    def test_always_cooperate(self):
        strategy = AlwaysCooperate()
        state = np.array([[1, 0, 1, 0]])  # Arbitrary state
        
        action = strategy.act(state)
        assert action == 0  # Always returns cooperate
    
    def test_always_defect(self):
        strategy = AlwaysDefect()
        state = np.array([[1, 0, 1, 0]])  # Arbitrary state
        
        action = strategy.act(state)
        assert action == 1  # Always returns defect
    
    def test_tit_for_tat(self):
        strategy = TitForTat()
        
        # First move should be cooperate
        first_state = np.array([[]])  # Empty history
        assert strategy.act(first_state) == 0
        
        # Should copy opponent's last move
        coop_state = np.array([[0]])  # Opponent cooperated
        assert strategy.act(coop_state) == 0
        
        defect_state = np.array([[1]])  # Opponent defected
        assert strategy.act(defect_state) == 1
    
    def test_random_strategy(self):
        strategy = RandomStrategy(seed=42)
        state = np.array([[]])
        
        # With fixed seed, check that we get deterministic sequence
        actions = [strategy.act(state) for _ in range(10)]
        
        # Make sure we have both actions
        assert 0 in actions
        assert 1 in actions


class TestModel:
    def test_logreg_shape(self):
        model = LogReg(input_dim=10)
        
        # Test single input
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 2)  # Logits for 2 actions
        
        # Test batch input
        x_batch = torch.randn(5, 10)
        output_batch = model(x_batch, batched=True)
        assert output_batch.shape == (5, 2)
    
    def test_mlp_shape(self):
        model = MLP(input_dim=10, hidden_dims=[20, 20])
        
        # Test single input
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 2)
        
        # Test batch input
        x_batch = torch.randn(5, 10)
        output_batch = model(x_batch, batched=True)
        assert output_batch.shape == (5, 2)
    
    def test_lstm_shape(self):
        model = LSTM(input_dim=2, hidden_dim=20)
        
        # Test single sequence input
        seq = torch.randn(1, 5, 2)  # 1 sequence, 5 timesteps, 2 features
        output = model(seq)
        assert output.shape == (1, 2)
        
        # Test batch input
        seq_batch = torch.randn(3, 5, 2)  # 3 sequences
        output_batch = model(seq_batch, batched=True)
        assert output_batch.shape == (3, 2)


class TestLearner:
    def test_policy_gradient_act(self):
        # Create a deterministic model for testing
        class DeterministicModel(torch.nn.Module):
            def forward(self, x, batched=False):
                # Always return logits that favor cooperation
                return torch.tensor([[2.0, 0.0]])
        
        model = DeterministicModel()
        learner = PolicyGradientLearner(model)
        
        # Test greedy action
        state = np.array([[0, 1, 0]])
        action = learner.act(state, exploration=False)
        assert action == 0  # Should select cooperation (highest logit)
        
        # Test with exploration disabled
        actions = [learner.act(state, exploration=False) for _ in range(10)]
        assert all(a == 0 for a in actions)  # Should always cooperate
    
    def test_policy_gradient_loss(self):
        model = LogReg(input_dim=2)
        learner = PolicyGradientLearner(model)
        
        # Create simple trajectory
        trajectory = Trajectory()
        trajectory.add(np.array([[0]]), 0, PAYOFF_MATRIX[0][0][0])  # State, action, reward (mutual cooperation)
        trajectory.add(np.array([[0, 0]]), 0, PAYOFF_MATRIX[0][0][0])
        
        # Test that loss calculation works
        loss = learner.loss([trajectory], gamma=0.99)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestTrajectory:
    def test_add_and_get(self):
        trajectory = Trajectory()
        
        # Add some transitions
        mutual_coop_reward = PAYOFF_MATRIX[0][0][0]
        mutual_defect_reward = PAYOFF_MATRIX[1][1][0]
        
        trajectory.add(np.array([[0]]), 0, mutual_coop_reward)
        trajectory.add(np.array([[0, 0]]), 1, mutual_defect_reward)
        
        # Check size
        assert len(trajectory) == 2
        
        # Check states
        states = trajectory.get_states()
        assert len(states) == 2
        assert np.array_equal(states[0], np.array([[0]]))
        assert np.array_equal(states[1], np.array([[0, 0]]))
        
        # Check actions
        actions = trajectory.get_actions()
        assert len(actions) == 2
        assert actions[0] == 0
        assert actions[1] == 1
        
        # Check rewards
        rewards = trajectory.get_rewards()
        assert len(rewards) == 2
        assert rewards[0] == mutual_coop_reward
        assert rewards[1] == mutual_defect_reward
    
    def test_discounted_rewards(self):
        trajectory = Trajectory()
        
        # Get values from payoff matrix
        coop_defect_reward = PAYOFF_MATRIX[0][1][0]
        mutual_coop_reward = PAYOFF_MATRIX[0][0][0]
        temptation_reward = PAYOFF_MATRIX[1][0][0]
        
        # Add transitions with rewards from payoff matrix
        trajectory.add(np.array([[0]]), 0, coop_defect_reward)
        trajectory.add(np.array([[0, 0]]), 0, mutual_coop_reward)
        trajectory.add(np.array([[0, 0, 0]]), 0, temptation_reward)
        
        # Get discounted sums with gamma=0.5
        # Expected: [r1 + 0.5*r2 + 0.25*r3, r2 + 0.5*r3, r3]
        gamma = 0.5
        discounted = trajectory.get_discounted_rewards(gamma=gamma)
        
        expected = np.array([
            coop_defect_reward + gamma*mutual_coop_reward + gamma*gamma*temptation_reward,
            mutual_coop_reward + gamma*temptation_reward,
            temptation_reward
        ])
        
        assert np.allclose(discounted, expected)


class TestEndToEndFlow:
    def test_environment_to_model_flow(self):
        """Test the flow of data from environment to model"""
        # Setup environment with known payoff matrix
        k = 3  # Memory size for state
        env = IPDEnvironment(payoffs=PAYOFF_MATRIX, max_steps=10, k=k)
        env.reset()
        
        # Play a sequence of moves to generate history
        # Both cooperate, then both defect, then mixed
        env.step(0, 0)  # Both cooperate
        env.step(1, 1)  # Both defect
        env.step(0, 1)  # Agent cooperates, opponent defects
        
        # Get state from environment
        state = env.get_state()
        
        # Verify state format
        assert state.shape[0] == 1  # Batch dimension
        assert state.shape[1] == 3  # k steps of history
        
        # Create model with input dimension matching flattened state
        input_dim = state.shape[1] * 2  # Each state has 2 values (agent and opponent actions)
        model = LogReg(input_dim=input_dim, output_dim=2)  # 2 actions: cooperate or defect
        
        # Test forward pass with state from environment
        logits = model(state)
        
        # Verify output format
        assert logits.shape == (1, 2)  # Should output logits for 2 actions
        
    def test_trajectory_to_model_flow(self):
        """Test the flow of data from trajectory to model during training"""
        # Create a trajectory with some known transitions
        k = 3
        trajectory = Trajectory()
        
        # Add transitions to trajectory (state, action, reward)
        trajectory.add(np.array([[0]]), 0, PAYOFF_MATRIX[0][0][0])  # Empty state -> cooperate
        trajectory.add(np.array([[0, 0]]), 0, PAYOFF_MATRIX[0][0][0])  # After mutual cooperation -> cooperate
        trajectory.add(np.array([[0, 0, 0]]), 1, PAYOFF_MATRIX[1][0][0])  # After two mutual cooperations -> defect
        
        # Get states from trajectory
        states = trajectory.get_states()
        
        # Verify states format
        assert isinstance(states, torch.Tensor)
        assert states.shape[0] == 3  # 3 transitions
        assert states.shape[2] == 2  # Each state has opponent and agent action
        
        # Flatten each state for LogReg model
        flattened_states = states.reshape(states.shape[0], -1)
        
        # Create model
        model = LogReg(input_dim=flattened_states.shape[1], output_dim=2)
        
        # Test forward pass with batch of states
        logits = model(flattened_states, batched=True)
        
        # Verify output format
        assert logits.shape == (3, 2)  # Should output logits for each state and each action
        
    def test_end_to_end_learning_flow(self):
        """Test the full flow from environment through trajectory to learning update"""
        # Setup components
        k = 3
        env = IPDEnvironment(payoffs=PAYOFF_MATRIX, max_steps=5, k=k)
        
        # Create a small model
        input_dim = 2 * k  # Each state has agent and opponent action over k timesteps
        model = LogReg(input_dim=input_dim, output_dim=2)
        
        # Setup learner
        learner = PolicyGradientLearner(model)
        
        # Create a simple opponent strategy
        opponent = TitForTat()
        
        # Simulate a training interaction
        trajectories = []
        
        # Play a game
        env.reset()
        history = []
        
        for _ in range(5):  # 5 steps per game
            state = env.get_state()
            action = learner.act(state, exploration=True)
            opponent_action = opponent.act(state)
            next_state, reward, done, _ = env.step(action, opponent_action)
            history.append((action, opponent_action, reward))
        
        # Create trajectory from game history
        trajectory = Trajectory()
        for state, action, reward in history:
            trajectory.add(state, action, reward)
        
        trajectories.append(trajectory)
        
        # Compute loss using the trajectory
        loss = learner.loss(trajectories, gamma=0.99)
        
        # Verify loss is a scalar tensor with gradient
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.shape == torch.Size([])  # Scalar

if __name__ == "__main__":
    pytest.main(["-xvs", "test_ipd.py"])