# Scaling Up Active Testing to Large Language Models

This is code for __Scaling Up Active Testing to Large Language Models__.

## Setup

You can set up a conda environment to use this repo as follows:

```
conda-env update -f environment.yml
conda activate at_llms
```

## Active Testing



### Model evaluation

To evaluate LLaMa-2 70B on one of the datasets, say SST-2:

```
device = 'cuda'
prompt_respond = f"""Classify the sentiment of the following sentence as "positive" or "negative". Respond with "positive" or "negative".\n"""
model_file = "llama2_70b"
dataset_file = "sst2"
```

Load model and dataset:

```
llama = model.Model(model_name="Llama-2-70b-chat",
                    model_file=model_file,
                    dataset_file=dataset_file,
                    prompt=prompt_respond,
                    device=device)
sst2_dataset = dataset_class.SST2Dataset('sst2')
sst2_dataset.load(llama.prompt, llama.model.tokenizer, device)
llama_evaluation = evaluation.EvaluationClassification(llama,
                                                        sst2_dataset,
                                                        device)
```

Model evaluation on a subset without in-context learning (make sure you have created the subset):

```
llama_evaluation.evaluation_procedure(subset_name, batch_size, save=True)
```

Model evaluation with in-context learning:

```
llama_evaluation.evaluation_icl_procedure(subset_name, nb_icl_examples, batch_size, save)
```

These should generate files containing model predictions for each label, as well as overall accuracy and loss.

### Acquisition

```
STEP = 1
RUNS = 3_000
SIZE = 400
```

To run I.I.D. acquisition:
```
llama_acquisition = acquisition.iidAcquisition(step=STEP,
                                                runs=RUNS,
                                                size=SIZE,
                                                eps=1e-15,
                                                model_file="llama2_70b",
                                                dataset_file='sst2',
                                                loss=metrics.cross_entropy_loss)
llama_acquisition.run_estimation()
```

To run active testing acquisition based on the cross-entropy between the surrogate model's and the target model's predictions:

```
llama_acquisition = acquisition.SurrogateEntropyAcquisition(step=STEP,
                                                            runs=RUNS,
                                                            size=SIZE,
                                                            eps=1e-15,
                                                            model_file="llama2_70b",
                                                            surrogate_file="llama2_70b_icl50",
                                                            dataset_file='sst2',
                                                            loss=metrics.cross_entropy_loss,
                                                            estimator=estimators.LUREEstimator,
                                                            cross_entropy=True,
                                                            nll=False,
                                                            temperature=None)
llama_acquisition.run_estimation(clip_percentage=0.1)
```

These should generate a file with active testing estimates at each acquisition step and for each run.

To change the acquisition function to predictive entropy of the surrogate model, set `cross_entropy=False`. To change the acquisition function to negative-log likelihood of the surrogate model, set `cross_entropy=False` and `nll=True`.

## Code structure

- `model.py` loads a model for evaluation.
- `dataset_class.py` loads a dataset for evaluation.
- `evaluation.py` runs model evaluation on a dataset.

Once you have evaluated a model, active testing experiments no longer need to load a model and a dataset and can be run without GPU. All the information you need is stored in `.pt` and `.npy` files.

- `estimators.py` contains estimator classes.
- `acquisition.py` enables I.I.D. and active testing acquisition.
- `ase.py` contains a class for Active Surrogate Estimators. Please refer to https://github.com/jlko/active-surrogate-estimators for original code.


