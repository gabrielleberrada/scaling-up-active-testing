import numpy as np
from tqdm import tqdm
from estimators import iidEstimator, LUREEstimator, ISEstimator
from metrics import entropy_loss
from scipy.special import softmax
import utils

class Acquisition:
    """Class for AT acquisition."""
    def __init__(self,
                step,
                runs,
                size,
                estimator,
                eps,
                model_file,
                dataset_file,
                loss):
        self.step = step
        self.runs = runs
        self.size = size
        self.estimator = estimator(loss)
        self.eps = eps
        self.model_file = model_file
        self.dataset_file = dataset_file
        self.rng = np.random.default_rng()

    def get_weights(self, scores, surrogate_scores, labels):
        raise NotImplementedError

    def acquire(self, pmf_distr):
        """Randomly sample from a distribution."""
        return self.rng.multinomial(1, pmf_distr).argmax()

    def get_pmf_distr(self, weights, clip_percentage):
        """Get distribution from weights."""
        if (weights < 0).sum() != 0:
            weights += weights.min()
        if weights.sum() != 0:
            weights = np.divide(weights, weights.sum())
        # does not affect uniform sampling
        weights = np.maximum(clip_percentage / len(weights), weights)
        weights = np.divide(weights, weights.sum())
        return weights

    def run(self, model_scores, surrogate_scores, labels, clip_percentage):
        """Run multiple AT evaluations."""
        weights = self.get_weights(model_scores, surrogate_scores, labels)
        set_size = len(model_scores)
        results = np.zeros((int(np.ceil(self.size/self.step)), self.runs))
        self.acquisition_weights = []
        self.samples = []
        for i in tqdm(range(self.runs)):
            remaining_idx = np.ones(set_size)
            acquisition_weights = np.zeros(self.size)
            samples_idx = []
            for m in range(self.size):
                pmf_distr = self.get_pmf_distr(weights[remaining_idx != 0], clip_percentage)
                new_idx = self.acquire(pmf_distr)
                corresponding_idx = np.where(remaining_idx == 1)[0][new_idx]
                remaining_idx[corresponding_idx] = 0
                samples_idx.append(corresponding_idx)
                acquisition_weights[m] = pmf_distr[new_idx]
                if m % self.step == 0:
                    results[m//self.step, i] = self.estimator.estimate(predicted_scores=model_scores[samples_idx],
                                                                       targets=labels[samples_idx],
                                                                       acquisition_weights=acquisition_weights[:m+1],
                                                                       set_size=set_size)
            self.acquisition_weights.append(acquisition_weights)
            self.samples.append(samples_idx)
        self.acquisition_weights = np.array(self.acquisition_weights)
        self.samples = np.array(self.samples)
        return results

    def run_bootstrap(self, model_scores, surrogate_scores, labels, clip_percentage):
        """Run bootstrap experiment based one acquisition run."""
        weights = self.get_weights(model_scores, surrogate_scores, labels)
        set_size = len(model_scores)
        results = np.zeros((int(np.ceil(self.size/self.step)), 1))
        remaining_idx = np.ones(set_size)
        acquisition_weights = np.zeros(self.size)
        samples_idx = []
        bootstrap_results = np.zeros((int(np.ceil(self.size/self.step)), self.runs))
        # single acquisition
        for m in range(self.size):
            pmf_distr = self.get_pmf_distr(weights[remaining_idx != 0], clip_percentage)
            new_idx = self.acquire(pmf_distr)
            corresponding_idx = np.where(remaining_idx == 1)[0][new_idx]
            remaining_idx[corresponding_idx] = 0
            samples_idx.append(corresponding_idx)
            acquisition_weights[m] = pmf_distr[new_idx]
            if m % self.step == 0:
                results[m//self.step] = self.estimator.estimate(predicted_scores=model_scores[samples_idx],
                                                               targets=labels[samples_idx],
                                                               acquisition_weights=acquisition_weights[:m+1],
                                                               set_size=set_size)
            # bootstrap evaluation
            sampled_scores = model_scores[samples_idx]
            sampled_labels = labels[samples_idx]
            for i in range(self.runs):
                samples = np.random.choice(np.arange(m+1), size=m+1, replace=True)
                bootstrap_results[m//self.step, i] = self.estimator.loss(sampled_scores[samples],
                                                                         sampled_labels[samples],
                                                                         self.estimator.weights[samples])
        return results, bootstrap_results, np.array(samples_idx)


class iidAcquisition(Acquisition):
    """Acquisition class for I.I.D. sampling."""
    def __init__(self,
                step,
                runs,
                size,
                eps,
                model_file,
                dataset_file,
                loss):
        super(iidAcquisition, self).__init__(step, runs, size, iidEstimator, eps, model_file, dataset_file, loss)
        self.saving_dir = f'{self.dataset_file}/{self.model_file}/{self.model_file}_iid'

    def get_weights(self, scores, surrogate_scores, labels):
        return np.ones(len(scores))

    def acquire(self, pool_size, n_samples):
        """We sample uniformly at once (without replacement) for faster implementation."""
        return self.rng.choice(pool_size, n_samples, replace=False)

    def run_estimation(self, subset_name='active', set_size=None, indices=None):
        # loading data
        model_scores = utils.load_tensors(f'{self.dataset_file}/{self.model_file}/{subset_name}_set_scores').numpy()[:set_size]
        labels = utils.load_tensors(f'{self.dataset_file}/{subset_name}_set_targets').numpy()[:set_size]
        if indices is not None:
            model_scores = model_scores[indices]
            labels = labels[indices]
        # running AT acquisition
        self.results = self.run(model_scores, surrogate_scores=None, labels=labels, clip_percentage=0.)
        # saving results
        add = '' if subset_name == 'active' else f'_{subset_name}'
        utils.save_arrays(self.results, f'{self.saving_dir}_loss{add}', add_duplicates=True)

    def run_bootstrap_estimation(self, clip_percentage=0.1, set_size=None, subset_name='active'):
        # loading data
        model_scores = utils.load_tensors(f'{self.dataset_file}/{self.model_file}/{subset_name}_set_scores').numpy()[:set_size]
        labels = utils.load_tensors(f'{self.dataset_file}/{subset_name}_set_targets').numpy()[:set_size]
        # running bootstrap acquisition
        self.results, self.bootstrap_results, self.samples = self.run_bootstrap(model_scores, surrogate_scores=None, 
                                                                                labels=labels, clip_percentage=clip_percentage)
        # saving results
        add = '' if subset_name == 'active' else f'_{subset_name}'
        utils.save_arrays(self.bootstrap_results, f'{self.saving_dir}_bootstrap_loss{add}', add_duplicates=True)
        utils.save_arrays(self.results, f'{self.saving_dir}_single_run_loss{add}', add_duplicates=True)
        utils.save_arrays(self.samples, f'{self.saving_dir}_single_run_samples{add}', add_duplicates=True)

    def get_coverage_estimation(self, n_runs, clip_percentage=0., set_size=None, subset_name='active'):
        model_scores = utils.load_tensors(f'{self.dataset_file}/{self.model_file}/{subset_name}_set_scores').numpy()[:set_size]
        labels = utils.load_tensors(f'{self.dataset_file}/{subset_name}_set_targets').numpy()[:set_size]
        self.estimated_loss = np.zeros((self.size, 0))
        self.estimated_variance = np.zeros((self.size, 0))
        self.samples = np.zeros((self.size, 0))
        for _ in tqdm(range(n_runs)):
            single_run_loss, bootstrap_loss, samples = self.run_bootstrap(model_scores, surrogate_scores=None, 
                                                                          labels=labels, clip_percentage=clip_percentage)
            bootstrap_variance = np.expand_dims(np.var(bootstrap_loss, axis=1), axis=1)
            self.estimated_loss = np.concatenate((self.estimated_loss, single_run_loss), axis=1)
            self.samples = np.concatenate((self.samples, np.expand_dims(samples, axis=1)), axis=1)
            self.estimated_variance = np.concatenate((self.estimated_variance, bootstrap_variance), axis=1)
        # saving results
        add = '' if subset_name == 'active' else f'_{subset_name}'
        utils.save_arrays(self.estimated_loss, f'{self.saving_dir}_at_loss{add}', add_duplicates=True)
        utils.save_arrays(self.samples, f'{self.saving_dir}_bootstrap_samples{add}', add_duplicates=True)
        utils.save_arrays(self.estimated_variance, f'{self.saving_dir}_bootstrap_variance{add}', add_duplicates=True)

    def run(self, model_scores, surrogate_scores, labels, clip_percentage):
        weights = self.get_weights(model_scores, surrogate_scores, labels)
        set_size = len(model_scores)
        results = np.zeros((int(np.ceil(self.size/self.step)), self.runs))
        self.acquisition_weights = []
        pmf_distr = self.get_pmf_distr(weights, clip_percentage)
        for i in tqdm(range(self.runs)):
            # sample at once
            samples_idx = self.acquire(set_size, self.size)
            acquisition_weights = pmf_distr[samples_idx]
            for m in range(self.size//self.step):
                results[m, i] = self.estimator.estimate(predicted_scores=model_scores[samples_idx[:m*self.step+1]],
                                                                    targets=labels[samples_idx[:m*self.step+1]],
                                                                    acquisition_weights=acquisition_weights[:m*self.step+1],
                                                                    set_size=set_size)
            self.acquisition_weights.append(acquisition_weights)
        self.acquisition_weights = np.array(self.acquisition_weights)
        return results

    def run_bootstrap(self, model_scores, surrogate_scores, labels, clip_percentage):
        weights = self.get_weights(model_scores, surrogate_scores, labels)
        set_size = len(model_scores)
        results = np.zeros((int(np.ceil(self.size/self.step)), 1))
        acquisition_weights = np.zeros(self.size)
        samples_idx = []
        # single run
        pmf_distr = self.get_pmf_distr(weights, clip_percentage)
        bootstrap_results = np.zeros((int(np.ceil(self.size/self.step)), self.runs))
        for m in tqdm(range(self.size)):
            new_idx = self.acquire(pmf_distr)
            samples_idx.append(new_idx)
            acquisition_weights[m] = pmf_distr[new_idx]
            if m % self.step == 0:
                results[m//self.step] = self.estimator.estimate(predicted_scores=model_scores[samples_idx],
                                                               targets=labels[samples_idx],
                                                               acquisition_weights=acquisition_weights[:m+1],
                                                               set_size=set_size)
            # bootstrap evaluation
            for i in range(self.runs):
                samples = np.random.choice(np.arange(m+1), size=m+1, replace=True)
                bootstrap_results[m//self.step, i] = self.estimator.loss(model_scores[samples_idx][samples],
                                                                         labels[samples_idx][samples],
                                                                         self.estimator.weights[samples])
        bootstrap_results = np.zeros((int(np.ceil(self.size/self.step)), self.runs))
        return results, bootstrap_results, np.array(samples_idx)

class TrueLossAcquisition(Acquisition):
    """Acquisition class based on the true loss distribution (optimal solution)."""
    def __init__(self,
                step,
                runs,
                size,
                eps,
                model_file,
                dataset_file,
                loss):
        super(TrueLossAcquisition, self).__init__(step, runs, size, LUREEstimator, eps, model_file, dataset_file, loss)
        self.saving_dir = f'{self.dataset_file}/{self.model_file}/{self.model_file}_optimal'

    def get_weights(self, scores, surrogate_scores, labels):
        predictions = softmax(scores, axis=1)
        predictions = np.clip(predictions, self.eps, 1/self.eps)
        return - (labels * np.log(predictions)).sum(axis=1)

    def run_estimation(self):
        # loading data
        model_scores = utils.load_tensors(f'{self.dataset_file}/{self.model_file}/active_set_scores').numpy()
        labels = utils.load_tensors(f'{self.dataset_file}/active_set_targets').numpy()
        # running AT acquisition
        self.results = self.run(model_scores, surrogate_scores=None, labels=labels, clip_percentage=0.)
        # saving results
        utils.save_arrays(self.results, f'{self.saving_dir}_loss')


class SelfEntropyAcquisition(Acquisition):
    """Acquisition class based on the predictive entropy of the target model.
    This is equivalent to sampling based on the cross-entropy between the surrogate model and the target model,
    where the surrogate model is the target model."""
    def __init__(self,
                step,
                runs,
                size,
                estimator,
                eps,
                model_file,
                dataset_file,
                loss,
                temperature=None):
        super(SelfEntropyAcquisition, self).__init__(step, runs, size, estimator, eps, model_file, dataset_file, loss)
        self.temperature = temperature
        if isinstance(self.estimator, iidEstimator):
            name = 'entropy'
            self.clip_percentage = 0.
            self.saving_dir = f'{self.dataset_file}/{self.model_file}/{self.model_file}_entropy'
        elif isinstance(self.estimator, LUREEstimator):
            name = 'self_loss'
            self.clip_percentage = 0.01
            self.saving_dir = f'{self.dataset_file}/{self.model_file}/{self.model_file}_{self.model_file}'
        else:
            raise NotImplementedError
    
    def get_weights(self, model_scores, surrogate_scores, labels):
        return entropy_loss(predictions=model_scores, 
                            temperature_model=self.temperature, 
                            surrogate_predictions=None, 
                            eps=self.eps)

    def run_estimation(self):
        # loading data
        model_scores = utils.load_tensors(f'{self.dataset_file}/{self.model_file}/active_set_scores').numpy()
        labels = utils.load_tensors(f'{self.dataset_file}/active_set_targets').numpy()
        # running AT acquisition
        self.results, self.weights, self.samples= self.run(model_scores, surrogate_scores=None, 
                                              labels=labels, clip_percentage=self.clip_percentage)
        # saving data
        utils.save_arrays(self.results, f'{self.saving_dir}_loss')


class SurrogateEntropyAcquisition(Acquisition):
    """General class for AT acquisition with a surrogate model."""
    def __init__(self,
                step,
                runs,
                size,
                eps,
                model_file,
                surrogate_file,
                dataset_file,
                loss,
                cross_entropy=True,
                nll=False,
                estimator=LUREEstimator,
                temperature=None):
        super(SurrogateEntropyAcquisition, self).__init__(step, runs, size, estimator, eps, model_file, dataset_file, loss)
        self.temperature = temperature
        self.surrogate_file = surrogate_file
        self.saving_dir = f'{self.dataset_file}/{self.model_file}/{self.model_file}_{self.surrogate_file}'
        self.cross_entropy = cross_entropy # if True, then CE
        self.nll = nll # applied if cross_entropy is False
        self.method = "" if estimator==LUREEstimator else "_is" if estimator==ISEstimator else "_iid"
    
    def get_weights(self, model_scores, surrogate_scores, labels):
        # defines the weights based on the acquisition function (CE, PE or NLL)
        if self.cross_entropy:
            return entropy_loss(predictions=model_scores, 
                                temperature_surrogate=self.temperature, 
                                surrogate_predictions=surrogate_scores, 
                                eps=self.eps)
        elif self.nll:
            return entropy_loss(predictions=surrogate_scores,
                                temperature_model=self.temperature,
                                surrogate_predictions=labels,
                                eps=self.eps)
        else:
            return entropy_loss(predictions=surrogate_scores,
                                temperature_model=self.temperature,
                                surrogate_predictions=None,
                                eps=self.eps)

    def run_estimation(self, clip_percentage=0.1, set_size=None, indices=None):
        # loading data
        model_scores = utils.load_tensors(f'{self.dataset_file}/{self.model_file}/active_set_scores').numpy()[:set_size]
        surrogate_scores = utils.load_tensors(f'{self.dataset_file}/{self.surrogate_file}/active_set_scores').numpy()[:set_size]
        labels = utils.load_tensors(f'{self.dataset_file}/active_set_targets').numpy()[:set_size]
        if indices is not None:
            model_scores = model_scores[indices]
            surrogate_scores = surrogate_scores[indices]
            labels = labels[indices]
        # running AT acquisition
        self.results = self.run(model_scores, surrogate_scores, labels, clip_percentage)
        # saving results
        post = '_nll' if self.nll else ('_entropy' if not(self.cross_entropy) else "")
        utils.save_arrays(self.results, f'{self.saving_dir}_loss{post}{self.method}', add_duplicates=True)

    def run_bootstrap_estimation(self, clip_percentage=0.01, set_size=None):
        # loading data
        model_scores = utils.load_tensors(f'{self.dataset_file}/{self.model_file}/active_set_scores').numpy()[:set_size]
        surrogate_scores = utils.load_tensors(f'{self.dataset_file}/{self.surrogate_file}/active_set_scores').numpy()[:set_size]
        labels = utils.load_tensors(f'{self.dataset_file}/active_set_targets').numpy()[:set_size]
        # running bootstrap acquisition
        self.results, self.bootstrap_results, self.samples = self.run_bootstrap(model_scores, surrogate_scores, labels, clip_percentage)
        # saving results
        post = '_nll' if self.nll else ('_entropy' if not(self.cross_entropy) else "")
        utils.save_arrays(self.bootstrap_results, f'{self.saving_dir}_bootstrap_loss{post}{self.method}', add_duplicates=True)
        utils.save_arrays(self.results, f'{self.saving_dir}_single_run_loss{post}{self.method}', add_duplicates=True)
        utils.save_arrays(self.samples, f'{self.saving_dir}_single_run_samples{post}{self.method}', add_duplicates=True)

    def get_coverage_estimation(self, n_runs, clip_percentage=0.01, set_size=None, subset_name='active'):
        # loading data
        model_scores = utils.load_tensors(f'{self.dataset_file}/{self.model_file}/{subset_name}_set_scores').numpy()[:set_size]
        surrogate_scores = utils.load_tensors(f'{self.dataset_file}/{self.surrogate_file}/active_set_scores').numpy()[:set_size]
        labels = utils.load_tensors(f'{self.dataset_file}/{subset_name}_set_targets').numpy()[:set_size]
        self.estimated_loss = np.zeros((self.size, 0))
        self.estimated_variance = np.zeros((self.size, 0))
        self.samples = np.zeros((self.size, 0))
        # compute coverage estimate
        for _ in tqdm(range(n_runs)):
            single_run_loss, bootstrap_loss, samples = self.run_bootstrap(model_scores, surrogate_scores, 
                                                                    labels, clip_percentage)
            bootstrap_variance = np.expand_dims(np.var(bootstrap_loss, axis=1), axis=1)
            self.estimated_loss = np.concatenate((self.estimated_loss, single_run_loss), axis=1)
            self.estimated_variance = np.concatenate((self.estimated_variance, bootstrap_variance), axis=1)
            self.samples = np.concatenate((self.samples, np.expand_dims(samples, axis=1)), axis=1)
        # saving results
        post = '_nll' if self.nll else ('_entropy' if not(self.cross_entropy) else "")
        utils.save_arrays(self.estimated_loss, f'{self.saving_dir}_at_loss{post}{self.method}', add_duplicates=True)
        utils.save_arrays(self.samples, f'{self.saving_dir}_bootstrap_samples{post}{self.method}', add_duplicates=True)
        utils.save_arrays(self.estimated_variance, f'{self.saving_dir}_bootstrap_variance{post}{self.method}', add_duplicates=True)