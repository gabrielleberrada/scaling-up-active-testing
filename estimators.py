import numpy as np

class Estimator:
    def __init__(self, loss):
        self.loss = loss

    def estimate(self, predicted_scores, targets, acquisition_weights, set_size):
        raise NotImplementedError

class iidEstimator(Estimator):
    """Subsample empirical risk (I.I.D. estimator) R_iid"""
    def __init__(self, loss):
        super(iidEstimator, self).__init__(loss)

    def estimate(self, predicted_scores, targets, acquisition_weights, set_size):
        return self.loss(predicted_scores, targets)

class LUREEstimator(Estimator):
    """Levelled Unbiased Risk Estimator."""
    def __init__(self, loss):
        super(LUREEstimator, self).__init__(loss)

    def estimate(self, predicted_scores, targets, acquisition_weights, set_size):
        M = len(predicted_scores)
        m = np.arange(1, M+1)
        if M == set_size:
            weights = np.ones(M)
        else:
            weights = 1 + (set_size - M)/(set_size - m) * (1 / ((set_size - m + 1) * acquisition_weights) - 1)
        self.weights = weights
        return self.loss(predicted_scores, targets, weights=weights)

class ISEstimator(Estimator):
    """Importance sampling estimator."""
    def __init__(self, loss):
        super(ISEstimator, self).__init__(loss)
    
    def estimate(self, predicted_scores, targets, acquisition_weights, set_size):
        return self.loss(predicted_scores, targets, weights=1/acquisition_weights/set_size)


        