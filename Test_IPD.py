import pytest
import numpy as np
import torch

from IPDEnvironment import IPDEnvironment
from Strategy import Strategy, TFT, Cu, Du, Random
from Model import LogReg, MLP, LSTM
from Learner import PolicyGradientLearner
from Trajectory import Trajectory
from Train import Trainer
from Optimizer import Optimizer

# should ALWAYS BE: MY ACTION, OPPONENT ACTION, MY REWARD, OPPONENT REWARD

COOPERATE = 0
DEFECT = 1
ACTIONS = [COOPERATE, DEFECT]
PAYOFF_MATRIX = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1)
}

class TestIPDEnvironment:
    def test_reset(self):
        k = 2
        env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds = 2, k = k)
        state = env.reset()
        assert env.current_step == 0
        assert len(env.history) == 0
        assert state.shape == (k, 2)  # empty history at start
    
    def test_step_first_move(self):
        k = 2   
        env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds = 2, k = k)
        env.reset()
        next_state, reward, _ = env.step(0, 0)  # both cooperate
        
        # check history
        assert len(env.history) == 1
        assert env.history[0] == (0, 0, PAYOFF_MATRIX[COOPERATE, COOPERATE][0], PAYOFF_MATRIX[COOPERATE, COOPERATE][1])
        
        # Check reward - should be R,R for mutual cooperation
        assert (reward, _) == PAYOFF_MATRIX[COOPERATE, COOPERATE]
        
        # Check state shape (first player gets history with opponent move)
        assert next_state.shape == (k, 2)
    
    def test_full_game(self):
        env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds = 5, k = 2)
        env.reset()
        
        # Play a full game
        rewards = []
        for _ in range(5):
            _, reward, opponent_reward = env.step(0, 0)  # Always cooperate
            rewards.append((reward, opponent_reward))
        
        assert len(rewards) == 5
        mutual_coop_reward = PAYOFF_MATRIX[COOPERATE, COOPERATE]
        assert all(r == mutual_coop_reward for r in rewards)  # All mutual cooperation
        assert len(env.history) == 5


class TestStrategies:
    def test_always_cooperate(self):
        strategy = Cu()
        state = np.array([[1, 0, 1, 0]])  # Arbitrary state
        
        action = strategy.act(state)
        assert action == 0  # Always returns cooperate
    
    def test_always_defect(self):
        strategy = Du()
        state = np.array([[1, 0, 1, 0]])  # Arbitrary state
        
        action = strategy.act(state)
        assert action == 1  # Always returns defect
    
    def test_tit_for_tat(self):
        strategy = TFT()
        env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds = 2, k = 2)
        env.reset()
        
        # First move should be cooperate
        first_state = env.get_state()  # Empty history
        assert strategy.act(first_state) == 0

        # Set opponent's last move to cooperate
        env.history.append((0, 0, PAYOFF_MATRIX[COOPERATE, COOPERATE][0], PAYOFF_MATRIX[COOPERATE, COOPERATE][1]))
        coop_state = env.get_state()  # Opponent cooperated
        assert strategy.act(coop_state) == 0
        
        # Set opponent's last move to defect
        env.history.append((1, 1, PAYOFF_MATRIX[DEFECT, DEFECT][0], PAYOFF_MATRIX[DEFECT, DEFECT][1]))
        defect_state = env.get_state()  # Opponent defected
        assert strategy.act(defect_state) == 1
    
    def test_random_strategy(self):
        strategy = Random()
        env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds = 2, k = 2)
        env.reset()
        
        # With fixed seed, check that we get deterministic sequence
        actions = [strategy.act(env.get_state()) for _ in range(10)]
        
        # Make sure we have both actions
        assert 0 in actions
        assert 1 in actions


class TestModel:
    def test_logreg_shape(self):
        model = LogReg(d_input=10, d_output=2)
        
        # Test single input
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 2)  # Logits for 2 actions
        
        # Test batch input
        x_batch = torch.randn(5, 10)
        output_batch = model(x_batch, batched=True)
        assert output_batch.shape == (5, 2)
    
    def test_mlp_shape(self):
        model = MLP(d_input=10, d_output=2, d_hidden=[20, 20])
        
        # Test single input
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 2)
        
        # Test batch input
        x_batch = torch.randn(5, 10)
        output_batch = model(x_batch, batched=True)
        assert output_batch.shape == (5, 2)
    
    def test_lstm_shape(self):
        model = LSTM(d_input=2, d_output=2, d_hidden=[20, 20])
        
        # Test single sequence input
        seq = torch.randn(1, 5, 2)  # 1 sequence, 5 timesteps, 2 features
        output = model(seq, batched = True)
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
                return torch.tensor([[100.0, 0.0]])
        
        model = DeterministicModel()
        learner = PolicyGradientLearner(model, device = "cpu", optimizer = None, terminal = False)
        
        # Test greedy action
        state = torch.tensor([[0, 1],[1, 0]])
        action = learner.act(state, epsilon = 0)
        assert action == 0
        
        # Test with full exploration
        actions = [learner.act(state, epsilon = 1) for _ in range(10)]
        assert all(a == 0 for a in actions) # hopefully! like most definitely
    
    def test_policy_gradient_loss(self):
        k = 4
        model = LogReg(d_input = k * 2, d_output = 2)
        learner = PolicyGradientLearner(model, device = "cpu", optimizer = None, terminal = False)
        
        # Create simple trajectory
        env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds = 10, k = k)
        env.reset()
        env.step(0, 0)
        env.step(1, 1)
        trajectory = Trajectory(env.history, k = k, my_payoff = env.payoff1, opponent_payoff = env.payoff2)
        
        # Test that loss calculation works
        loss = learner.loss([trajectory], gamma=0.99)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

class TestTrajectory:
    def test_add_from_history(self):
        history = []
        # my action, their action, my reward, their reward
        history.append((0, 0, PAYOFF_MATRIX[COOPERATE, COOPERATE][0], PAYOFF_MATRIX[COOPERATE, COOPERATE][1]))
        history.append((0, 1, PAYOFF_MATRIX[COOPERATE, DEFECT][0], PAYOFF_MATRIX[COOPERATE, DEFECT][1]))
        history.append((1, 0, PAYOFF_MATRIX[DEFECT, COOPERATE][0], PAYOFF_MATRIX[DEFECT, COOPERATE][1]))
        history.append((1, 1, PAYOFF_MATRIX[DEFECT, DEFECT][0], PAYOFF_MATRIX[DEFECT, DEFECT][1]))

        trajectory = Trajectory(history, k = 2, my_payoff = 0, opponent_payoff = 0)
        
        # check states
        states = trajectory.get_states()
        assert len(states) == 4
        
        # manual check of states
        assert np.array_equal(states[0], np.array([[2, 2],[2, 2]]))
        assert np.array_equal(states[1], np.array([[2, 2],[0, 0]]))
        assert np.array_equal(states[2], np.array([[0, 0],[0, 1]]))
        assert np.array_equal(states[3], np.array([[0, 1],[1, 0]]))
        
        # check actions
        actions = trajectory.get_actions()
        assert len(actions) == 4
        assert actions[0] == 0
        assert actions[1] == 0
        assert actions[2] == 1
        assert actions[3] == 1
    
    def test_add_from_env(self):
        env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds = 10, k = 2)
        env.reset()
        env.step(0, 0)
        env.step(1, 1)
        env.step(0, 1)
        env.step(0, 0)
        env.step(1, 1)
        env.step(1, 0)

        trajectory = Trajectory(env.history, k = 3, my_payoff = env.payoff1, opponent_payoff = env.payoff2)
        assert len(trajectory.rewards) == 6
        assert len(trajectory.actions) == 6

        # check states
        states = trajectory.get_states()
        assert len(states) == 6
        assert np.array_equal(states[0], np.array([[2, 2],[2, 2], [2, 2]]))
        assert np.array_equal(states[1], np.array([[2, 2], [2, 2], [0, 0]]))
        assert np.array_equal(states[2], np.array([[2, 2],[0, 0],[1, 1]])) 
        assert np.array_equal(states[3], np.array([[0, 0],[1, 1],[0, 1]]))
        assert np.array_equal(states[4], np.array([[1, 1],[0, 1],[0, 0]]))
        assert np.array_equal(states[5], np.array([[0, 1],[0, 0],[1, 1]]))

        # check actions
        actions = trajectory.get_actions()
        assert len(actions) == 6
        
    
    def test_discounted_rewards(self):
        env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds = 3, k = 3)
        env.reset()
        
        # get (player 1) values from payoff matrix
        coop_defect_reward = PAYOFF_MATRIX[COOPERATE, DEFECT][0]
        mutual_coop_reward = PAYOFF_MATRIX[COOPERATE, COOPERATE][0]
        temptation_reward = PAYOFF_MATRIX[DEFECT, COOPERATE][0]
        
        # my action, opponent action
        env.step(0, 1)
        env.step(0, 0)
        env.step(1, 0)
        
        # create trajectory
        trajectory = Trajectory(env.history, k = 3, my_payoff = env.payoff1, opponent_payoff = env.payoff2)

        # sanity check discounted sums
        gamma = 0.5
        discounted = trajectory.get_reward_sums(gamma=gamma)
        
        expected = torch.tensor([
            coop_defect_reward + gamma*mutual_coop_reward + gamma*gamma*temptation_reward,
            mutual_coop_reward + gamma*temptation_reward,
            temptation_reward
        ])
        
        assert torch.allclose(discounted, expected)


class TestEndToEndFlow:
    def test_environment_to_model_flow(self):
        """Test the flow of data from environment to model"""
        k = 10  # Memory size for state
        env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds = 10, k = k)
        
        # play a sequence of moves to generate history
        # both cooperate, then both defect, then mixed
        env.step(0, 0) 
        env.step(1, 1)
        env.step(0, 1)
        env.step(0, 0)
        env.step(1, 1)
        
        # get state from environment
        state = env.get_state()
        
        # verify state format
        assert state.shape[0] == k  
        assert state.shape[1] == 2  
        
        input_dim = 2 * k  # each state has 2 values (agent and opponent actions)
        model = LogReg(d_input=input_dim, d_output = 2)
        
        # test forward pass with state from environment
        logits = model(state)
        
        # verify output format
        assert logits.shape == (1, 2)  # should output logits for 2 actions

        trajectory = Trajectory(env.history, k = k, my_payoff = env.payoff1, opponent_payoff = env.payoff2)
        states = trajectory.get_states()
        actions = trajectory.get_actions()
        rewards = trajectory.get_reward_sums()
        
        logits = model(states, batched = True)
        action_log_probs = torch.gather(logits, dim = 1, index = actions)
        
        expected = torch.tensor([logits[i, actions[i]] for i in range(len(actions.flatten()))])
        assert torch.isclose(action_log_probs.flatten(), expected).all()
        
        # verify states format
        assert isinstance(states, torch.Tensor)
        assert states.shape[0] == 5
        assert states.shape[2] == 2
        
        # try batched forward pass
        logits = model(states, batched=True)
        
        # verify output format
        assert logits.shape == (5, 2)  # Should output logits for each state and each action

if __name__ == "__main__":
    pytest.main(["-xvs", "Test_IPD.py"])