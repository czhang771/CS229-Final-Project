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

SCHEDULERS = {
    'step': optim.lr_scheduler.StepLR,
    'multistep': optim.lr_scheduler.MultiStepLR,
    'exponential': optim.lr_scheduler.ExponentialLR,
    'cosine': optim.lr_scheduler.CosineAnnealingLR,
    'plateau': optim.lr_scheduler.ReduceLROnPlateau,
    'cyclic': optim.lr_scheduler.CyclicLR,
    'onecycle': optim.lr_scheduler.OneCycleLR,
    'cosine_restart': optim.lr_scheduler.CosineAnnealingWarmRestarts,
}

class OptimizerWithScheduler:
    def __init__(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()
    
    def scheduler_step(self, metric=None):
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metric)
            else:
                self.scheduler.step()
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class Optimizer:
    @classmethod
    def create(cls, model_params, name, scheduler_type=None, scheduler_params=None, **kwargs):
        """Wrapper to create a PyTorch optimizer by name with given parameters."""
        name = name.lower()
        
        if name not in OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{name}' not recognized. Available optimizers: "
                f"{', '.join(OPTIMIZERS.keys())}"
            )
        
        optimizer_class = OPTIMIZERS[name]
        optimizer = optimizer_class(model_params, **kwargs)
        
        scheduler = None
        if scheduler_type is not None:
            if scheduler_type not in SCHEDULERS:
                raise ValueError(
                    f"Scheduler '{scheduler_type}' not recognized. Available schedulers: "
                    f"{', '.join(SCHEDULERS.keys())}"
                )
            
            scheduler_class = SCHEDULERS[scheduler_type]
            scheduler_params = scheduler_params or {}
            scheduler = scheduler_class(optimizer, **scheduler_params)
        
        return OptimizerWithScheduler(optimizer, scheduler)


def create_optimizer(model, name, params_dict=None):
    """ Create an optimizer for a model using a name and parameter dictionary. """
    if params_dict is None:
        params_dict = {}
    
    # Extract scheduler parameters if provided
    scheduler_type = params_dict.pop('scheduler_type', None)
    scheduler_params = params_dict.pop('scheduler_params', None)
    
    return Optimizer.create(
        model.parameters(), 
        name, 
        scheduler_type=scheduler_type,
        scheduler_params=scheduler_params,
        **params_dict
    )