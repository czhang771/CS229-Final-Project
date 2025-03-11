import yaml
import os
import torch
import json
import itertools
from Train import Trainer
from Model import MLP, LSTM, LogReg
from Learner import PolicyGradientLearner, ActorCriticLearner
from IPDEnvironment import IPDEnvironment
from Strategy import *

PAYOFF_MATRIX = {
    (0, 0): (3, 3),
    (0, 1): (0, 5),
    (1, 0): (5, 0),
    (1, 1): (1, 1),
}

RESULTS_DIR = "results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_config(config_path="config.yaml"):
    """Load experiment settings from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def create_learner(learner_type, actor_model, critic_model_type, device, optimizer, lr, scheduler, scheduler_params):
    """Creates the learner based on experiment config."""
    param_dict = {"lr": lr, "scheduler_type": scheduler, "scheduler_params": scheduler_params}

    if learner_type == "policy_gradient":
        return PolicyGradientLearner(actor_model, device, optimizer, terminal=False, param_dict=param_dict)

    elif learner_type == "actor_critic":
        critic_model = create_model(critic_model_type, actor_model.d_input, actor_model.d_output, actor_model.hidden_sizes)
        return ActorCriticLearner(actor_model, critic_model, device, optimizer, optimizer, terminal=False, 
                                  param_dict={"actor": param_dict, "critic": param_dict})
    else:
        raise ValueError(f"Unsupported learner type: {learner_type}")
    
def create_model(model_type, input_size, output_size, hidden_layers):
    """Creates model based on config."""
    if model_type == "MLP":
        return MLP(input_size, output_size, hidden_layers)
    elif model_type == "LSTM":
        return LSTM(input_size, output_size, hidden_layers)
    elif model_type == "LogReg":
        return LogReg(input_size, output_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def create_opponent_combinations(config):
    """
    Generate all possible opponent sets of fixed size.

    Args:
        config (dict): Experiment config dictionary.

    Returns:
        list[list[str]]: List of all possible opponent combinations.
    """
    all_strategies = config["opponents"]["training"]["available_strategies"]
    num_opps = config["opponents"]["training"]["num_opponents"]

    # Generate all possible subsets of size `num_opps`
    opponent_combinations = list(itertools.combinations(all_strategies, num_opps))

    return [list(combo) for combo in opponent_combinations]

def create_opponents(opponent_names, config):
    """
    Creates a list of opponent strategy instances based on provided names.

    Args:
        opponent_names (list[str]): List of opponent strategy names.
        config (dict): Experiment config dictionary.

    Returns:
        list[Strategy]: List of instantiated opponent strategies.
    """
    strategy_map = {
        "Cu": Cu(),
        "Du": Du(),
        "Random": Random(),
        "Cp": Cp(config["strategy_params"].get("Cp_p", 0.5)),  
        "TFT": TFT(),
        "ImpTFT": ImpTFT(config["strategy_params"].get("ImpTFT_p", 0.9)),
        "GTFT": GTFT(
            R=config["strategy_params"].get("GTFT_R", 3),
            P=config["strategy_params"].get("GTFT_P", 1),
            T=config["strategy_params"].get("GTFT_T", 5),
            S=config["strategy_params"].get("GTFT_S", 0),
        ),
        "GRIM": GRIM(),
        "WSLS": WSLS(),
    }

    opponents = []
    for name in opponent_names:
        if name in strategy_map:
            opponents.append(strategy_map[name])
        else:
            raise ValueError(f"Opponent strategy '{name}' not found in strategy_map.")
    
    return opponents

def run_all_experiments():
    """Runs all experiments with dynamically generated opponent combinations of fixed size."""
    config = load_config()
    env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds=config["training"]["num_games"], k=config["training"]["k"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = config["training"]["k"] * 2
    output_size = 2  

    opponent_combinations = create_opponent_combinations(config)  

    param_combinations = itertools.product(
        config["hyperparameters"]["learning_rates"],
        config["hyperparameters"]["optimizers"],
        config["hyperparameters"]["scheduler_types"],
        config["hyperparameters"]["scheduler_params"]["gamma"],
        opponent_combinations  
    )

    for idx, (lr, optimizer, scheduler, gamma, opponent_list) in enumerate(param_combinations):
        actor_model = create_model(config["model"]["actor_model"], input_size, output_size, config["model"]["hidden_layers"])
        learner = create_learner(config["training"]["learner_type"], actor_model, config["model"]["critic_model"], device, optimizer, lr, scheduler, {"gamma": gamma})
        opponent_mix = create_opponents(opponent_list, config)

        trainer = Trainer(env, learner, opponent_mix, k=config["training"]["k"])
        trainer.train_MC(epochs=config["training"]["epochs"], num_games=config["training"]["num_games"], game_length=config["training"]["game_length"])

        # Save results
        save_name = f"{config['experiment_name']}_L{config['training']['learner_type']}_A{config['model']['actor_model']}_C{config['model']['critic_model']}_LR{lr}_OPT{optimizer}_SCH{scheduler}_OPP{''.join(opponent_list)}"
        model_path = f"{RESULTS_DIR}/{save_name}.pth"
        torch.save(learner.model.state_dict(), model_path)

        results = {
            "learner_type": config["training"]["learner_type"],
            "learning_rate": lr,
            "scheduler": scheduler,
            "optimizer": optimizer,
            "actor_model_type": config["model"]["actor_model"],
            "critic_model_type": config["model"]["critic_model"],
            "opponents": opponent_list,
            "final_score": trainer.score_history[-1],
            "loss_history": trainer.loss_history
        }
        json.dump(results, open(f"{RESULTS_DIR}/{save_name}.json", "w"))

        print(f"Finished experiment: {save_name}")

if __name__ == "__main__":
    run_all_experiments()