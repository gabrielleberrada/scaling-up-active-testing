from huggingface_model import HuggingfaceModel
import os
import utils
import torch

class Model:
    """Class to load model."""
    def __init__(self,
                model_name,
                model_file,
                dataset_file,
                prompt,
                device):
        self.model_name = model_name
        self.model_file = model_file
        self.dataset_file = dataset_file
        if not(os.path.isdir(self.dataset_file)):
            os.mkdir(self.dataset_file)
        if not(os.path.isdir(f'{self.dataset_file}/{self.model_file}')):
            os.mkdir(f'{self.dataset_file}/{self.model_file}')
        self.random_seed = 1
        self.debug = None
        self.stop_sequences = ["<s>"]
        self.max_new_tokens = 1
        self.temperature = 1.
        self.top_k = None
        self.top_p = None
        self.device = device
        self.prompt = prompt
        self.load_model()

    def load_model(self):
        self.model = HuggingfaceModel(name=self.model_name, 
                                     random_seed=self.random_seed, 
                                     debug=self.debug, 
                                     stop_sequences=self.stop_sequences, 
                                     max_new_tokens=self.max_new_tokens, 
                                     temperature=self.temperature, 
                                     top_k=self.top_k, 
                                     top_p=self.top_p)
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.tokenizer.padding_side = 'left'

    def get_path(self, filename):
        return f'{self.dataset_file}/{self.model_file}/{filename}'

    def predict(self, text):
        self.model.predict(text)

    def get_tokens(self, words):
        if os.path.isfile(self.get_path('tokens.pt')):
            self.tokens = utils.load_tensors(self.get_path('tokens'))
        else:
            tokens = []
            for word in words:
                tokens.append(self.model.tokenizer(word, return_tensors='pt')['input_ids'][:, -1].reshape(1))
            self.tokens = torch.concat(tokens)
            utils.save_tensors(self.tokens, self.get_path('tokens'))

    
        