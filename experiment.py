import utils
import estimators

class Experiment:
    """Class to run acquisition experiments."""
    def __init__(self,
                 model,
                 dataset,
                 surrogate,
                 acquisition,
                 clip_percentage):
        self.model = model
        self.dataset = dataset
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.clip_percentage = clip_percentage
        self.dir_path = f'{self.dataset.filename}/{self.model.filename}'
        if isinstance(self.acquisition, acquisition.iidAcquisition):
            self.filename = f'{self.model.filename}_iid_loss'
        elif isinstance(self.acquisition, acquisition.TrueLossAcquisition):
            self.filename = f'{self.model.filename}_optimal_loss'
        elif isinstance(self.acquisition, acquisition.SelfEntropyAcquisition):
            if isinstance(self.acquisition.estimator, estimators.iidEstimator):
                self.filename = f'{self.model.filename}_entropy_loss'
            else:
                self.filename = f'{self.model.filename}_self_loss'
        else:
            self.filename = f'{self.model.filename}_{self.surrogate.filename}_loss'
            
    def compute_evaluation_results(self):
        pass

    def get_evaluation_results(self, set_name):
        model_scores = utils.load(f'{self.dataset.filename}/{self.model.filename}/{set_name}_set_scores.pt')
        surrogate_scores = utils.load(f'{self.dataset.filename}/{self.surrogate.filename}/{set_name}_set_scores.pt')
        labels = utils.load(f'{self.dataset.filename}/{set_name}_set_targets.pt')
        return model_scores, surrogate_scores, labels

    def run_estimation(self, set_name):
        model_scores, surrogate_scores, labels = self.get_evaluation_results(set_name)
        self.results = self.acquisition.run(model_scores, surrogate_scores, labels, self.clip_percentage)
        utils.save_tensors(self.results, f'{self.dir_path}/{self.filename}')

    