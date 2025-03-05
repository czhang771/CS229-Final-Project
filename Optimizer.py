import torch
import torch.optim as optim

OPTIMIZERS = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'lbfgs': optim.LBFGS,
        'sparseadam': optim.SparseAdam,
        'rprop': optim.Rprop
    }

class Optimizer:
    @classmethod
    def create(cls, model_params, name, **kwargs) -> torch.optim.Optimizer:
        """Wrapper to create a PyTorch optimizer by name with given parameters."""
        
        name = name.lower()

        if name not in OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{name}' not recognized. Available optimizers: "
                f"{', '.join(cls.OPTIMIZERS.keys())}"
            )
        
        optimizer_class = OPTIMIZERS[name]
        return optimizer_class(model_params, **kwargs)


def create_optimizer(model, name, params_dict=None):
    """ Create an optimizer for a model using a name and parameter dictionary. """
    if params_dict is None:
        params_dict = {}
    
    return Optimizer.create(model.parameters(), name, **params_dict)