import torch
import numpy as np
import itertools
import json
from IPDEnvironment import IPDEnvironment
from Model import MLP, LSTM
from Learner import PolicyGradientLearner, ActorCriticLearner
from Train import Trainer
from Strategy import *  
from tqdm import tqdm  # For progress bar

# Define hyperparameter search space
learning_rates = [0.001, 0.005, 0.01]
scheduler_gammas = [0.9, 0.9, 0.99, 0.999]
seeds = [42, 123, 999]  # Multiple seeds for averaging

policy_gradient_models = [MLP, LSTM]
actor_critic_models = [
    ("MLP", "MLP"),
    ("LSTM", "LSTM"),
    ("MLP", "LSTM"),
    ("LSTM", "MLP")
]

# Fixed parameters
NUM_ACTIONS = 2
STATE_DIM = 2
k = 5
num_rounds = 1000
epochs = 50
num_games = 10
game_length = 20
batch_size = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAYOFF_MATRIX = {
    (0, 0): (3, 3),
    (0, 1): (0, 5),
    (1, 0): (5, 0),
    (1, 1): (1, 1),
}
total_runs = len(learning_rates) * len(scheduler_gammas) * len(seeds) * (len(policy_gradient_models) + len(actor_critic_models))
progress_bar = tqdm(total=total_runs, desc="Hyperparameter Tuning Progress")

# Store results
results = {}
for lr, gamma in itertools.product(learning_rates, scheduler_gammas):
    for model_class in policy_gradient_models:
        scores = []
        losses = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)


            # Setup environment and opponent
            env = IPDEnvironment(PAYOFF_MATRIX, num_rounds, k)
            opponent = TFT()

            # Create model
            if model_class == MLP:
                model = MLP(d_input= STATE_DIM * k, d_output=NUM_ACTIONS, d_hidden=[4 * k, 4 * k])
            elif model_class == LSTM:
                model = LSTM(d_input=STATE_DIM, d_output=NUM_ACTIONS, d_hidden=[8 * STATE_DIM, 4 * STATE_DIM])

            # Policy Gradient Learner
            learner = PolicyGradientLearner(model, device, "adamw", terminal=False,
                                            param_dict={"lr": lr, "scheduler_type": "exponential",
                                                        "scheduler_params": {"gamma": gamma}})
            trainer = Trainer(env, learner, opponent, k=k, gamma=0.99)
            trainer.train_MC(epochs=epochs, game_length=game_length, num_games=num_games)

            scores.append(np.mean(trainer.score_history))
            losses.append(np.mean(trainer.loss_history))

        results[f"PG_{model_class.__name__}_LR{lr}_G{gamma}"] = {
            "avg_score": np.mean(scores),
            "avg_loss": np.mean(losses)
        }

        key = f"PG_{model_class.__name__}_LR{lr}_G{gamma}"
        results[key] = {
            "avg_score": np.mean(scores),
            "avg_loss": np.mean(losses)
        }

        # Print & Save Progress
        print(f"Saving: {key} → {results[key]}")
        with open("hyperparameter_results.json", "w") as f:
            json.dump(results, f, indent=4)


    for actor_name, critic_name in actor_critic_models:
        scores = []
        losses = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Setup environment and opponent
            env = IPDEnvironment(PAYOFF_MATRIX, num_rounds, k)
            opponent = TFT()

            # Instantiate models
            actor = MLP(d_input=STATE_DIM * k, d_output=NUM_ACTIONS, d_hidden=[4 * k, 4 * k]) if actor_name == "MLP" else LSTM(d_input=STATE_DIM, d_output=NUM_ACTIONS, d_hidden=[4 * STATE_DIM, 8 * STATE_DIM, 4 * STATE_DIM])
            critic = MLP(d_input=STATE_DIM * k, d_output=NUM_ACTIONS, d_hidden=[4 * k, 4 * k]) if critic_name == "MLP" else LSTM(d_input=STATE_DIM, d_output=NUM_ACTIONS, d_hidden=[4 * STATE_DIM, 8 * STATE_DIM, 4 * STATE_DIM])

            learner = ActorCriticLearner(actor, critic, device, 
                                         actor_optimizer="adamw", critic_optimizer="adamw",
                                         terminal=False,
                                         param_dict={"actor": {"lr": lr, "scheduler_type": "exponential", "scheduler_params": {"gamma": gamma}},
                                                     "critic": {"lr": lr, "scheduler_type": "exponential", "scheduler_params": {"gamma": gamma}}})

            trainer = Trainer(env, learner, opponent, k=k, gamma=0.99)
            trainer.train_AC(epochs=epochs, game_length=game_length, num_games=num_games, batch_size=batch_size)

            scores.append(np.mean(trainer.score_history))
            losses.append(np.mean([x[0] + x[1] for x in trainer.loss_history]))  # Sum actor and critic loss

        key = f"AC_{actor_name}_{critic_name}_LR{lr}_G{gamma}"
        results[key] = {
            "avg_score": np.mean(scores),
            "avg_loss": np.mean(losses)
        }

        # Print & Save Progress
        print(f"Saving: {key} → {results[key]}")
        with open("hyperparameter_results.json", "w") as f:
            json.dump(results, f, indent=4)


progress_bar.close()  # Close progress bar when done
print("Hyperparameter tuning complete. Results saved.")