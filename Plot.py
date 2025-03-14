import json
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

# global constants
AGENT_TOPLINE = 61

def load_results(results_base: str, 
                num_opponents: int = 13, 
                ext_string: str = "_evaluated.json",
                fields: Optional[List[str]] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    load results from a single json file containing entries for various n values
    
    args:
        results_base: base name of results file
        num_opponents: maximum number of opponents to consider
        ext_string: extension string of the json file
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
    
    # load the single json file
    filepath = f"results/{results_base}{ext_string}"
    try:
        consolidated_data = json.load(open(filepath))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"error: could not load file {filepath}: {e}")
        return results_dict
    
    # group results by number of opponents
    for i in range(1, num_opponents + 1):
        # filter entries for this number of opponents
        n_entries = {}
        for key, data in consolidated_data.items():
            # check if this entry is for the current number of opponents (i)
            if f"_numOpp{i}_" in key or (isinstance(data.get('opponents', []), list) and len(data.get('opponents', [])) == i):
                n_entries[key] = data
        
        if not n_entries:
            # no entries found for this number of opponents
            continue
            
        # calculate statistics for each field
        for field in fields:
            # extract values for this field from all entries with this n
            values = [entry[field] for entry in n_entries.values() if field in entry]
            if values:
                # calculate mean and standard deviation
                results_dict[field][0][i-1] = np.mean(values)
                results_dict[field][1][i-1] = np.std(values)
    
    return results_dict

def plot_experiment_results(ax, experiment_configs: List[Dict],
                          title: str = "Experiment Results",
                          x_label: str = "number of opponents",
                          save_path: Optional[str] = None,
                          shared_axis: bool = False,
                          show_legend: bool = False) -> None:
    """
    plot results from multiple experiments
    
    args:
        experiment_configs: list of experiment configuration dictionaries 
                           each with keys:
                           - 'results_base': base filename
                           - 'ext_string': extension string (optional)
                           - 'fields': list of fields to plot (optional)
                           - 'labels': optional field labels for legend
                           - 'colors': optional colors for each field
                           - 'secondary_axis': fields to plot on secondary axis
                           - 'y_label': labels for primary and secondary axes
        title: plot title
        x_label: x-axis label
        save_path: path to save the figure, if None, figure is displayed
        shared_axis: whether to use a shared y-axis for all fields
        show_legend: whether to show the legend on this plot
    """
    # if we need separate axes for different metrics
    axes = [ax]
    if not shared_axis and any('secondary_axis' in config for config in experiment_configs):
        ax2 = ax.twinx()
        axes.append(ax2)
    
    for config in experiment_configs:
        results_base = config['results_base']
        ext_string = config.get('ext_string', "_evaluated.json")
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
            label = f"{field_labels.get(field, field)}"
            color = colors.get(field, None)
            
            # plot the line
            line = current_ax.plot(x, avg_values, label=label, color=color)
            
            # use the line color for consistency if no specific color was given
            fill_color = color if color is not None else line[0].get_color()
            
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
    
    # set labels and title directly on the axis
    ax.set_xlabel(x_label)
    ax.set_title(title)
    
    # add legends to each axis with non-empty content
    if show_legend:
        for ax_idx, current_ax in enumerate(axes):
            handles, labels = current_ax.get_legend_handles_labels()
            if handles:
                loc = 'upper right' if ax_idx == 1 else 'upper right'
                current_ax.legend(loc=loc)

# example usage
if __name__ == "__main__":
    results = {
        "PG_LogReg": {
            "score": 28.5,
            "steps": 30.5
        },
        "PG_MLP": {
            "score": 29.9,
            "steps": 32.9
        },
        "PG_LSTM": {
            "score": 28.1,
            "steps": 28.4
        },
        "AC_LR_LR": {
            "score": 20.72,
            "steps": 35.1
        },
        "AC_MLP_MLP": {
            "score": 19.51,
            "steps": 20.9
        },
        "AC_MLP_LSTM": {
            "score": 19.22,
            "steps": 29.6
        }
    }

    # create a 2 x 3 plot (PG as the first row, AC as the second row)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # make fonts bigger
    plt.rcParams.update({'font.size': 16})
    # Define the order of models to match 2x3 layout
    pg_models = ["PG_LogReg", "PG_MLP", "PG_LSTM"]
    ac_models = ["AC_LR_LR", "AC_MLP_MLP", "AC_MLP_LSTM"]
    all_models = [pg_models, ac_models]
    
    for row_idx, model_row in enumerate(all_models):
        for col_idx, key in enumerate(model_row):
            # Determine if this is the top right plot (for legend)
            is_top_right = (row_idx == 0 and col_idx == 2)
            
            plot_experiment_results(axs[row_idx, col_idx], [
                {
                    'results_base': key,
                    'ext_string': "_evaluated.json",
                    'fields': ["scores"],
                    'labels': {"scores": "out-of-domain score"}
                }
            ], title=key, show_legend=is_top_right)
            
            # Add horizontal line with label
            line = axs[row_idx, col_idx].axhline(results[key]["score"], color='black', linestyle='--')
            
            # Add label to horizontal line (only on top right for legend)
            if is_top_right:
                line.set_label("in-domain score")
                axs[row_idx, col_idx].legend(loc='upper right')
            
            # Set y-axis bounds to be the same for all plots
            axs[row_idx, col_idx].set_ylim(0, 50)
    
    # Ensure consistent formatting across all subplots
    for ax in axs.flat:
        ax.set_xlabel('n opponents')
    
    # Fix layout
    plt.tight_layout()
    #plt.show()
    plt.savefig("all_results.pdf")