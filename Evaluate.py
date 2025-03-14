from Model import LSTM, MLP, LogReg
from IPDEnvironment import IPDEnvironment
import Strategy
import torch
import Train
import json
import yaml
import numpy as np
from Learner import ActorCriticLearner, PolicyGradientLearner
import Experiment2
k = 5
STATE_DIM = 2
NUM_ACTIONS = 2

def load_model(model_path):
    """Load model from path"""
    model_name = model_path.split('_')[1] # get model name
    if model_name == "LSTM":
        model = LSTM(d_input = STATE_DIM, d_output = NUM_ACTIONS, d_hidden = [8 * STATE_DIM, 4 * STATE_DIM])
    elif model_name == "MLP":
        model = MLP(d_input = STATE_DIM * k, d_output = NUM_ACTIONS, d_hidden = [4 * k, 4 * k])
    elif model_name == "LR":
        model = LogReg(d_input = STATE_DIM * k, d_output = NUM_ACTIONS)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    model.load_state_dict(torch.load(model_path))
    return model


def evaluate_model(results_stem, path, num_games, game_length):
    """Evaluate model against strong opponent"""
    # use details from results_stem config file
    # with open(f"configs/{results_stem}.yaml", 'r') as f:
    #     config = yaml.safe_load(f)
    
    # optimizer = config['hyperparameters']['optimizers']
    # model = load_model(path)
    # if results_stem.split('_')[0] == "AC":
    #     # critic model doesn't matter
    #     model = Experiment2.create_ac_learner(model, config['model']['critic_model'], "cpu", optimizer, config['hyperparameters']['actor_lr'], config['hyperparameters']['critic_lr'], config['hyperparameters']['scheduler_types'], config['hyperparameters']['scheduler_params'], k)
    # else:
    #     model = Experiment2.create_pg_learner(model, "cpu", optimizer, config['hyperparameters']['actor_lr'], config['hyperparameters']['scheduler_types'], config['hyperparameters']['scheduler_params'])
    
    # random initialization
    # model = PolicyGradientLearner(MLP(d_input = STATE_DIM * k, d_output = NUM_ACTIONS, d_hidden = [4 * k, 4 * k]), device = "cpu", optimizer_name = "adamw", param_dict = {"lr": 0.01})
    model = Strategy.Du()
    opponent = [Strategy.AdaptiveMemoryStrategy()]
    # opponent = [Strategy.Strong()]
    env = IPDEnvironment(payoff_matrix = Train.PAYOFF_MATRIX, num_rounds = 100, k = k)
    trainer = Train.Trainer(env, model, opponent, k = k, gamma = 0.99, random_threshold = 0.5, min_epsilon = 0.1)
    scores = trainer.evaluate(game_length = game_length, num_games = num_games, verbose = False)
    env.print_game_sequence()
    
    return scores

def evaluate_all_models(results_stem):
    """Evaluate all models in results_path"""
    new_results = {}

    for i in range(1, 14):
        results_path = f"results/{results_stem}_numOpp{i}_CURRICULUMTrue.json"
        with open(results_path, 'r') as f:
            results = json.load(f)

        for result in results:
            new_results[result] = results[result]
            new_results[result]['scores'] = evaluate_model(results_stem = results_stem, path = results[result]['model_path'], num_games = 1, game_length = 20)
            # save to json
            print(new_results[result]['scores'])
        
        json.dump(new_results, open(f"results/{results_stem}_evaluated_llm.json", 'w'))

if __name__ == "__main__":
    scores = []
    for i in range(100):
        score = evaluate_model(None, None, num_games = 1, game_length = 20)
        scores.append(score)
    
    print(np.mean(scores), np.std(scores))
    # evaluate_all_models("PG_LSTM")
