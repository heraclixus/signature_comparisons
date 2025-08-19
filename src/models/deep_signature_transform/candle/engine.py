import torch
import torch.nn.utils as nnutils
from enum import Enum
from typing import Callable, Any, Dict, Optional, Union
import torch.nn.functional as F


class Events(Enum):
    """Event enum for engine state transitions"""
    STARTED = "started"
    EPOCH_STARTED = "epoch_started"
    ITERATION_COMPLETED = "iteration_completed"
    EPOCH_COMPLETED = "epoch_completed"
    COMPLETED = "completed"


class State:
    """Engine state container"""
    def __init__(self):
        self.iteration = 0
        self.epoch = 0
        self.output = None
        self.metrics = {}
        # For compatibility with existing code
        self.pbar = None


class Engine:
    """Simple training engine to replace ignite.Engine"""
    
    def __init__(self, process_function):
        self.process_function = process_function
        self.state = State()
        self.event_handlers = {event: [] for event in Events}
    
    def on(self, event: Events):
        """Decorator to register event handlers"""
        def decorator(handler):
            self.event_handlers[event].append(handler)
            return handler
        return decorator
    
    def _fire_event(self, event: Events):
        """Fire all handlers for a given event"""
        for handler in self.event_handlers[event]:
            handler(self)
    
    def run(self, data_loader, max_epochs=1):
        """Run the training loop"""
        self._fire_event(Events.STARTED)
        
        for epoch in range(max_epochs):
            self.state.epoch = epoch + 1
            self._fire_event(Events.EPOCH_STARTED)
            
            for batch in data_loader:
                self.state.iteration += 1
                self.state.output = self.process_function(self, batch)
                self._fire_event(Events.ITERATION_COMPLETED)
            
            self._fire_event(Events.EPOCH_COMPLETED)
        
        self._fire_event(Events.COMPLETED)


class Evaluator:
    """Simple evaluator to replace ignite evaluator"""
    
    def __init__(self, model, device=None, metrics=None):
        self.model = model
        self.device = device
        self.metrics = metrics or {}
        self.state = State()
    
    def run(self, data_loader):
        """Run evaluation on data loader"""
        self.model.eval()
        
        # Reset metrics
        metric_values = {name: [] for name in self.metrics.keys()}
        
        with torch.no_grad():
            for batch in data_loader:
                x, y = _prepare_batch(batch, device=self.device)
                y_pred = self.model(x)
                
                # Compute metrics
                for name, metric in self.metrics.items():
                    if hasattr(metric, 'compute'):
                        # For custom metric objects
                        metric_values[name].append(metric.compute(y_pred, y))
                    else:
                        # For loss functions
                        metric_values[name].append(metric(y_pred, y).item())
        
        # Average the metrics
        self.state.metrics = {
            name: sum(values) / len(values) 
            for name, values in metric_values.items()
        }


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training/evaluation"""
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
    else:
        x, y = batch, None
    
    if device is not None:
        if x is not None:
            x = x.to(device, non_blocking=non_blocking)
        if y is not None:
            y = y.to(device, non_blocking=non_blocking)
    
    return x, y


def create_supervised_evaluator(model, device=None, metrics=None):
    """Create a supervised evaluator"""
    return Evaluator(model, device=device, metrics=metrics)


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch,
                              check_nan=False,
                              grad_clip=None,
                              output_predictions=False):
    """Create a supervised trainer without ignite dependency.
    
    Performs training with optional:
    - NaN checking on predictions
    - Gradient clipping  
    - Recording predictions made by a model

    Arguments:
        model: The model to train
        optimizer: The optimizer to use
        loss_fn: Loss function
        device: Device to use ('cuda' or 'cpu')
        non_blocking: Whether to use non-blocking transfer to device
        prepare_batch: Function to prepare batches (default handles (x,y) tuples)
        check_nan: Whether to check predictions for NaN values (default False)
        grad_clip: Value to clip gradient norm to, or None for no clipping
        output_predictions: Whether to return (loss, predictions) or just loss
    """

    if device:
        model.to(device)

    if grad_clip is False:
        grad_clip = None
    elif grad_clip is True:
        grad_clip = 1.0

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)

        if check_nan and torch.isnan(y_pred).any():
            e = RuntimeError('Model generated NaN value.')
            e.y = y
            e.y_pred = y_pred
            e.x = x
            e.model = model
            raise e

        loss = loss_fn(y_pred, y)
        loss.backward()

        if grad_clip is not None:
            nnutils.clip_grad_norm_(model.parameters(), grad_clip, norm_type='inf')

        optimizer.step()

        if output_predictions:
            return loss.item(), y_pred
        else:
            return loss.item()

    return Engine(_update)


# Metric classes to replace ignite metrics
class MeanSquaredError:
    """MSE metric compatible with ignite interface"""
    
    def compute(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true).item()


class Loss:
    """Loss metric wrapper compatible with ignite interface"""
    
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
    
    def compute(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true).item()
