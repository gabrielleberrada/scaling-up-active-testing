from estimators import iidEstimator
import utils
from metrics import entropy_loss, softmax_clamp
import torch
import numpy as np

class ActiveSurrogateEstimator:
    """Class for ASE"""
    def __init__(self,
                model_file,
                surrogate_file,
                dataset_file,
                loss,
                predictive_entropy=False,
                nll=False,
                temperature=None
                ):
        self.model_file = model_file
        self.surrogate_file = surrogate_file
        self.dataset_file = dataset_file
        self.loss = loss
        self.estimator = iidEstimator(loss)
        # default is CE, switch predictive_entropy or nll to True to change
        self.predictive_entropy = predictive_entropy
        self.nll = nll
        self.temperature = temperature
        self.eps = 1e-15
        self.rng = np.random.default_rng()

    def get_weights(self, model_scores, surrogate_scores, labels):
        if self.predictive_entropy:
            return entropy_loss(predictions=surrogate_scores,
                                temperature_model=self.temperature,
                                surrogate_predictions=None,
                                eps=self.eps)
        elif self.nll:
            return entropy_loss(predictions=surrogate_scores,
                                temperature_model=self.temperature,
                                surrogate_predictions=labels,
                                eps=self.eps)
        else:
            return entropy_loss(predictions=model_scores, 
                                temperature_surrogate=self.temperature, 
                                surrogate_predictions=surrogate_scores, 
                                eps=self.eps)
            

    def acquire(self, pmf_distr):
        return self.rng.multinomial(1, pmf_distr).argmax()

    def get_pmf_distr(self, weights, clip_percentage=0.):
        if (weights < 0).sum() != 0:
            weights += weights.min()
        if weights.sum() != 0:
            weights = np.divide(weights, weights.sum())
        # does not affect uniform sampling
        weights = np.maximum(clip_percentage / len(weights), weights)
        weights = np.divide(weights, weights.sum())
        return weights

    def get_estimate(self, model_scores, surrogate_scores, labels, samples_idx):
        """Obtain ASE based on sampled indices samples_idx"""
        if self.predictive_entropy:
            predictions = surrogate_scores
            predictions[samples_idx] = model_scores[samples_idx]
            targets = softmax_clamp(surrogate_scores)
            targets[samples_idx] = labels[samples_idx]
        elif self.nll:
            predictions = surrogate_scores
            predictions[samples_idx] = model_scores[samples_idx]
            targets = labels
        else:
            predictions = model_scores
            targets = softmax_clamp(surrogate_scores)
            targets[samples_idx] = labels[samples_idx]
        return self.estimator.estimate(predicted_scores=predictions,
                                       targets=targets,
                                       acquisition_weights=None,
                                       set_size=None)
    
    def zero_shot_estimate(self, subset_name='active', set_size=None):
        """Zero-shot estimate: the surrogate model is not updated."""
        model_scores = utils.load_tensors(f'{self.dataset_file}/{self.model_file}/{subset_name}_set_scores')[:set_size]
        surrogate_scores = utils.load_tensors(f'{self.dataset_file}/{self.surrogate_file}/{subset_name}_set_scores')[:set_size]
        softmax = torch.nn.Softmax(dim=1)
        if self.temperature:
            preds = model_scores
            scores = softmax(surrogate_scores/self.temperature)
        else:
            preds = model_scores
            scores = softmax(surrogate_scores)
        preds = preds.numpy()
        scores = scores.numpy()
        return self.estimator.estimate(predicted_scores=preds, 
                                      targets=scores, 
                                      acquisition_weights=None, 
                                      set_size=None)

                                   
        