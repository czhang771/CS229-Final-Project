import json
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

N = 12
# plot curriculum results
def load_results(results_base):
    # load all results from /results folder from results_base_numOpp{i}.json for i in range(1, 13)
    avg_steps = np.zeros(N)
    std_steps = np.zeros(N)
    avg_scores = np.zeros(N)
    std_scores = np.zeros(N)
    for i in range(1, N + 1):
        results = json.load(open(f"results/{results_base}_numOpp{i}.json"))
        avg_steps[i - 1] = np.mean([results[key]["steps_to_convergence"] for key in results])
        std_steps[i - 1] = np.std([results[key]["steps_to_convergence"] for key in results])
        avg_scores[i - 1] = np.mean([results[key]["score_history"] for key in results])
        std_scores[i - 1] = np.std([results[key]["score_history"] for key in results])
    
    return avg_steps, std_steps, avg_scores, std_scores

def plot_results(results_base):
    x = np.arange(1, N + 1)
    avg_steps, std_steps, avg_scores, std_scores = load_results(results_base)
    # print with shading for 1 std
    plt.plot(x, avg_steps, label = "Steps to Convergence")
    plt.fill_between(x, avg_steps - std_steps, avg_steps + std_steps, alpha = 0.2)
    plt.plot(x, avg_scores, label = "Average Score")
    plt.fill_between(x, avg_scores - std_scores, avg_scores + std_scores, alpha = 0.2)
    plt.legend()
    plt.xlabel('Number of Opponents')
    plt.ylabel('Average Score')
    plt.title(f'{results_base}, randomized opponents')
    plt.show()

plot_results("AC_LR_LR")
