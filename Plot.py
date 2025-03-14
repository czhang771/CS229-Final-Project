import json
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import glob
import re

# global constants
AGENT_TOPLINE = 61

def load_results(results_base: str, 
                num_opponents: int = 13, 
                ext_string: str = '_numOpp{i}_CURRICULUMTrue.json',
                fields: Optional[List[str]] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    load results from json files, supporting two formats:
    1. single file per n value: results/{results_base}_numOpp{i}.json
    2. single file with all n values: results/{results_base}{ext_string}
    
    args:
        results_base: base name of results files
        num_opponents: number of opponents to load results for
        ext_string: extension string format with {i} placeholder for opponent number
        fields: list of fields to extract from results, defaults to ["steps_to_convergence", "score_history"]
    
    returns:
        dictionary mapping field names to tuples of (average, std_deviation) arrays
    """
    # default fields to extract if none provided
    if fields is None:
        fields = ["steps_to_convergence", "score_history"]
    
    # initialize result arrays
    results_dict = {}
    for field in fields:
        results_dict[field] = (np.zeros(num_opponents), np.zeros(num_opponents))
    
    # check if we have consolidated file
    consolidated_filename = f"results/{results_base}{ext_string.format(i='*')}"
    consolidated_files = glob.glob(consolidated_filename.replace('*', ''))
    
    if consolidated_files:
        # Case 1: We have a single consolidated file with all results
        consolidated_data = json.load(open(consolidated_files[0]))
        
        # Group results by number of opponents
        for i in range(1, num_opponents + 1):
            # Filter entries for this number of opponents
            n_entries = {}
            for key, data in consolidated_data.items():
                # Look for numOpp in the key or in the 'opponents' field length
                if f"_numOpp{i}_" in key or (isinstance(data.get('opponents', []), list) and len(data.get('opponents', [])) == i):
                    n_entries[key] = data
            
            if not n_entries:
                continue
                
            # Calculate statistics for each field
            for field in fields:
                values = [entry[field] for entry in n_entries.values() if field in entry]
                if values:
                    results_dict[field][0][i-1] = np.mean(values)
                    results_dict[field][1][i-1] = np.std(values)
    else:
        # Case 2: Separate file for each n value
        for i in range(1, num_opponents + 1):
            # Try different filename patterns
            possible_files = [
                f"results/{results_base}_numOpp{i}.json",
                f"results/{results_base}_numOpp{i}_CURRICULUMTrue.json",
                f"results/{results_base}{ext_string.format(i=i)}"
            ]
            
            result_data = None
            for filepath in possible_files:
                try:
                    result_data = json.load(open(filepath))
                    break
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
            
            if result_data is None:
                print(f"warning: could not find results file for {results_base} with n={i}")
                continue
            
            # Extract each requested field
            for field in fields:
                try:
                    values = [result_data[key][field] for key in result_data if field in result_data[key]]
                    if values:
                        results_dict[field][0][i-1] = np.mean(values)
                        results_dict[field][1][i-1] = np.std(values)
                except (KeyError, TypeError):
                    print(f"warning: could not extract field {field} for n={i}")
    
    return results_dict

def plot_experiment_results(experiment_configs: List[Dict],
                          title: str = "Experiment Results",
                          x_label: str = "number of opponents",
                          save_path: Optional[str] = None,
                          shared_axis: bool = False) -> None:
    """
    plot results from multiple experiments
    
    args:
        experiment_configs: list of experiment configuration dictionaries 
                           each with keys:
                           - 'results_base': base filename
                           - 'ext_string': extension format string (optional)
                           - 'fields': list of fields to plot (optional)
                           - 'labels': optional field labels for legend
                           - 'colors': optional colors for each field
                           - 'secondary_axis': fields to plot on secondary axis
                           - 'y_label': labels for primary and secondary axes
        title: plot title
        x_label: x-axis label
        save_path: path to save the figure, if None, figure is displayed
        shared_axis: whether to use a shared y-axis for all fields
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # if we need separate axes for different metrics
    axes = [ax]
    if not shared_axis and any('secondary_axis' in config for config in experiment_configs):
        ax2 = ax.twinx()
        axes.append(ax2)
    
    for config in experiment_configs:
        results_base = config['results_base']
        ext_string = config.get('ext_string', '_numOpp{i}_CURRICULUMTrue.json')
        num_opponents = config.get('num_opponents', 13)
        fields = config.get('fields', ["steps_to_convergence", "score_history"])
        field_labels = config.get('labels', {})
        colors = config.get('colors', {})
        
        # x values (number of opponents)
        x = np.arange(1, num_opponents + 1)
        
        # load results for this experiment
        results = load_results(results_base, num_opponents, ext_string, fields)
        
        # plot each field
        for field in fields:
            avg_values, std_values = results[field]
            
            # determine which axis to use
            axis_idx = 1 if config.get('secondary_axis', {}).get(field, False) else 0
            current_ax = axes[axis_idx]
            
            # get label and color for this field
            label = f"{results_base}: {field_labels.get(field, field)}"
            color = colors.get(field, None)
            
            # plot the line
            line = current_ax.plot(x, avg_values, label=label, color=color)
            
            # use the line color for consistency if no specific color was given
            fill_color = color if color else line[0].get_color()
            
            # plot the standard deviation band
            current_ax.fill_between(
                x, 
                avg_values - std_values, 
                avg_values + std_values, 
                alpha=0.2, 
                color=fill_color
            )
            
            # set y label if provided
            if 'y_label' in config:
                if axis_idx == 0:
                    axes[0].set_ylabel(config['y_label'].get('primary', ''))
                elif len(axes) > 1:
                    axes[1].set_ylabel(config['y_label'].get('secondary', ''))
    
    # set labels and title
    plt.xlabel(x_label)
    
    # add legends to each axis with non-empty content
    for ax_idx, current_ax in enumerate(axes):
        handles, labels = current_ax.get_legend_handles_labels()
        if handles:
            loc = 'upper right' if ax_idx == 1 else 'upper left'
            current_ax.legend(loc=loc)
    
    plt.title(title)
    plt.tight_layout()
    
    # save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# example usage
if __name__ == "__main__":
    # example 1: single experiment with multiple metrics on different axes
    # plot_experiment_results([
    #     {
    #         'results_base': "AC_MLP_MLP_more",
    #         'fields': ["steps_to_convergence", "score_history"],
    #         'labels': {
    #             "steps_to_convergence": "steps to convergence",
    #             "score_history": "average score"
    #         },
    #         'secondary_axis': {"score_history": True},
    #         'y_label': {
    #             'primary': 'steps to convergence',
    #             'secondary': 'average score'
    #         }
    #     }
    # ], title="AC_MLP_MLP_more, scheduled opponents")
    
    # # example 2: multiple experiments comparing steps to convergence
    # plot_experiment_results([
    #     {'results_base': "AC_MLP_MLP", 'fields': ["steps_to_convergence"]},
    #     {'results_base': "AC_LR_LR", 'fields': ["steps_to_convergence"]},
    #     {'results_base': "AC_MLP_LSTM", 'fields': ["steps_to_convergence"]}
    # ], title="Comparison of Steps to Convergence", save_path="AC_steps_to_convergence.pdf")
    
    # example 3: custom extension string
    plot_experiment_results([
        {
            'results_base': "AC_MLP_LSTM",
            'ext_string': "_evaluated.json",
            'fields': ["steps_to_convergence", "score_history", "scores"]
        }
    ])