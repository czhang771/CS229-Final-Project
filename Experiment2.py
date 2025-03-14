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
NUM_ACTIONS = 2
STATE_DIM = 2
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
        return yaml.safe_load(file), config_path

def create_pg_learner(actor_model, device, optimizer, lr, scheduler, scheduler_params):
    param_dict = {"lr": lr, "scheduler_type": scheduler, "scheduler_params": scheduler_params}
    return PolicyGradientLearner(actor_model, device, optimizer, terminal=False, param_dict=param_dict)



def create_ac_learner(actor_model, critic_model_type, device, optimizer, actor_lr, critic_lr, scheduler, scheduler_params, k):
    """Creates the learner based on experiment config."""
    critic_model = create_model(critic_model_type, k)
    return ActorCriticLearner(actor_model, critic_model, device, optimizer, optimizer, terminal=False, 
                                  param_dict={"actor": {"lr": actor_lr, "scheduler_type": scheduler, "scheduler_params": scheduler_params},
                                                        "critic": {"lr": critic_lr, "scheduler_type": scheduler, "scheduler_params": scheduler_params}})

   
    
def create_model(model_type, k):
    """Creates model based on config."""
    if model_type == "MLP":
        return MLP(STATE_DIM * k, NUM_ACTIONS, [4 * k, 4 *k])
    elif model_type == "LSTM":
        return LSTM(STATE_DIM, NUM_ACTIONS, [8 * STATE_DIM, 4 * STATE_DIM])
    elif model_type == "LogReg":
        return LogReg(STATE_DIM * k, NUM_ACTIONS)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def create_opponent_combinations(config, num_opps):
    """
    Generate all possible opponent sets of fixed size.

    Args:
        config (dict): Experiment config dictionary.

    Returns:
        list[list[str]]: List of all possible opponent combinations.
    """
    all_strategies = config["opponents"]["training"]["available_strategies"]
    # num_opps = config["opponents"]["training"]["num_opponents"]
    num_samples = config["opponents"]["training"]["num_samples"]

    # Generate all possible subsets of size `num_opps`
    all_combinations = list(itertools.combinations(all_strategies, num_opps))

    # Ensure we don't sample more than possible combinations
    num_samples = min(num_samples, len(all_combinations))

    # Randomly sample `num_samples` unique combinations
    sampled_combinations = random.sample(all_combinations, num_samples)

    return [list(combo) for combo in sampled_combinations]

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
        "Cp": Cp(config["strategy_params"].get("Cp_p", 0.7)),  
        "TFT": TFT(),
        "STFT": STFT(), 
        "GTFT": GTFT(
            R=config["strategy_params"].get("GTFT_R", 3),
            P=config["strategy_params"].get("GTFT_P", 1),
            T=config["strategy_params"].get("GTFT_T", 5),
            S=config["strategy_params"].get("GTFT_S", 0),
        ),
        "GrdTFT": GrdTFT(),
        "ImpTFT": ImpTFT(config["strategy_params"].get("ImpTFT_p", 0.95)),
        "TFTT": TFTT(),
        "TTFT": TTFT(),
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

def run_all_experiments(config_name):
    """Runs all experiments with dynamically generated opponent combinations of fixed size."""
    config, config_filename = load_config(config_name)
    env = IPDEnvironment(payoff_matrix=PAYOFF_MATRIX, num_rounds=config["training"]["num_games"], k=config["training"]["k"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = config["training"]["k"]
    all_results = {}

    for num_opponents in range(1, 14):
        opponent_combinations = create_opponent_combinations(config, num_opponents)  

        actor_model = create_model(config["model"]["actor_model"], k)
        optimizer = config["hyperparameters"]["optimizers"]
        scheduler = config["hyperparameters"]["scheduler_types"]
        scheduler_params = config["hyperparameters"]["scheduler_params"]
        
        if config["training"]["learner_type"] == "policy_gradient":
            learner = create_pg_learner(actor_model, device, optimizer, config["hyperparameters"]["actor_lr"], scheduler, scheduler_params)
        elif config["training"]["learner_type"] == "actor_critic":
            learner = create_ac_learner(
                actor_model, config["model"]["critic_model"], device, optimizer,
                config["hyperparameters"]["actor_lr"], config["hyperparameters"]["critic_lr"],
                scheduler, scheduler_params, k
            )

        for idx, opponent_list in enumerate(opponent_combinations):
            opponent_mix = create_opponents(opponent_list, config)
            trainer = Trainer(env, learner, opponent_mix, k=config["training"]["k"], gamma=0.99)
            opp_string = "_".join(opponent_list)  
            print(opp_string)
            
            if config["training"]["learner_type"] == "policy_gradient":
                """
                trainer.train_MC(epochs=int(config["training"]["epochs"]),
                                game_length=int(config["training"]["game_length"]),
                                num_games=int(config["training"]["num_games"]),
                                entropy_coef=0)
                """
                
                #trainer.train_MC(int(config["training"]["epochs"]),int(config["training"]["game_length"]),int(config["training"]["num_games"]))
            
                trainer.train_MC(50, 20, 10)
            elif config["training"]["learner_type"] == "actor_critic":
                trainer.train_AC(epochs=int(config["training"]["epochs"]), 
                                game_length=int(config["training"]["game_length"]),
                                num_games=int(config["training"]["num_games"]), 
                                batch_size=int(config["training"]["batch_size"]))


            # Save results
            opp_string = "_".join(opponent_list)  
            save_name = f"{config['experiment_name']}_OPP{opp_string}"

            model_path = f"{RESULTS_DIR}/{save_name}.pth"

            #saves model
            torch.save(learner.model.state_dict(), model_path)

            all_results[save_name] = {
                "opponents": opponent_list,
                "final_score": trainer.score_history[-1],
                "loss_history": trainer.loss_history,
                "model_path": model_path
            }

            print(f"Finished experiment: {save_name}")

        json_filename = f"{RESULTS_DIR}/{config['experiment_name']}_numOpp{num_opponents}.json"
        with open(json_filename, "w") as f:
            json.dump(all_results, f, indent=4)

        print(f"All results for {num_opponents} opponents saved to: {json_filename}")


if __name__ == "__main__":
    run_all_experiments("AC.yaml")