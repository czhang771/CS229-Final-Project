from Model import LSTM, MLP, LogReg
import IPDEnvironment
import Strategy
import torch
import Train
import json

k = 5
STATE_DIM = 2
NUM_ACTIONS = 2

def load_model(model_path):
    """Load model from path"""
    model_name = model_path.split('_')[1] # get model name
    if model_name == "LSTM":
        model = LSTM(d_input = STATE_DIM, d_output = NUM_ACTIONS, d_hidden = [4 * STATE_DIM, 8 * STATE_DIM, 4 * STATE_DIM])
    elif model_name == "MLP":
        model = MLP(d_input = STATE_DIM * k, d_output = NUM_ACTIONS, d_hidden = [4 * k, 4 * k])
    elif model_name == "LR":
        model = LogReg(d_input = STATE_DIM * k, d_output = NUM_ACTIONS)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    model.load_state_dict(torch.load(model_path))
    return model


def evaluate_model(path, num_games, game_length):
    """Evaluate model against strong opponent"""
    model = load_model(path)
    opponent = Strategy.Strong()
    env = IPDEnvironment(payoff_matrix = Train.PAYOFF_MATRIX, num_rounds = num_games, k = k)
    trainer = Train.Trainer(env, model, opponent, k = k, gamma = 0.99, random_threshold = 0.5, min_epsilon = 0.1)
    scores = trainer.evaluate(game_length, num_games)
    return scores

def evaluate_all_models(results_path):
    """Evaluate all models in results_path"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    for result in results:
        result['scores'] = evaluate_model(path =result['model_path'], num_games = 1, game_length = result['game_length'])
    
    json.dump(results, open(results_path + '_evaluated.json', 'w'))
