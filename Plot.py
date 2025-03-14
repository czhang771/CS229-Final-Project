import json
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

AGENT_TOPLINE = 61

N = 13
# plot curriculum results
def load_results(results_base):
    # load all results from /results folder from results_base_numOpp{i}.json for i in range(1, 13)
    avg_steps = np.zeros(N)
    std_steps = np.zeros(N)
    avg_scores = np.zeros(N)
    std_scores = np.zeros(N)
    for i in range(1, N + 1):
        try:
            results = json.load(open(f"results/{results_base}_numOpp{i}_CURRICULUMTrue.json"))
        except FileNotFoundError:
            results = json.load(open(f"results/{results_base}_numOpp{i}.json"))
        avg_steps[i - 1] = np.mean([results[key]["steps_to_convergence"] for key in results])
        std_steps[i - 1] = np.std([results[key]["steps_to_convergence"] for key in results])
        avg_scores[i - 1] = np.mean([results[key]["score_history"] for key in results])
        std_scores[i - 1] = np.std([results[key]["score_history"] for key in results])
    
    return avg_steps, std_steps, avg_scores, std_scores

def plot_single_experiment_results(results_base):
    x = np.arange(1, N + 1)
    avg_steps, std_steps, avg_scores, std_scores = load_results(results_base)
    # print with shading for 1 std
    # do separate left and right y-axis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, avg_steps, label = "Steps to Convergence")
    ax1.set_ylim(0, 100)
    ax1.fill_between(x, avg_steps - std_steps, avg_steps + std_steps, alpha = 0.2)
    ax2.plot(x, avg_scores, label = "Average Score")
    ax2.set_ylim(0, 100)
    ax2.fill_between(x, avg_scores - std_scores, avg_scores + std_scores, alpha = 0.2)
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')
    plt.xlabel('Number of Opponents')
    ax1.set_ylabel('Steps to Convergence')
    ax2.set_ylabel('Average Score')
    plt.title(f'{results_base}, scheduled opponents')
    plt.show()

def plot_multiple_experiment_steps(results = ["AC_MLP_MLP", "AC_LR_LR", "AC_MLP_LSTM"]):
    # make pretty colormap all blue but exclude white
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.5, 0.8, len(results)))
    x = np.arange(1, N + 1)
    for i, result in enumerate(results):
        avg_steps, std_steps, avg_scores, std_scores = load_results(result)
        # plot with shading for 1 std
        plt.plot(x, avg_steps, label = result, color = colors[i])
        plt.fill_between(x, avg_steps - std_steps, avg_steps + std_steps, alpha = 0.2, color = colors[i])
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('Steps to Convergence')
    plt.show()

plot_single_experiment_results("AC_MLP_MLP_more")
plot_multiple_experiment_steps()
