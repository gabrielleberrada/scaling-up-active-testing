import os
import torch
import numpy as np

def save_tensor_(result, filename, i=0, add_duplicates=False):
    if add_duplicates:
        v_format = f'_v{i}' if i > 0 else ''
        if os.path.isfile(f'{filename}{v_format}.pt'):
            save_tensor_(result.cpu(), f'{filename}', i=i+1, add_duplicates=True)
        else:
            if i == 0:
                torch.save(result.cpu(), f'{filename}.pt')
            else:
                print(f'Warning: Saved as version {i}')
                torch.save(result.cpu(), f'{filename}_v{i}.pt')
    else:
        assert not(os.path.isfile(f'{filename}.pt')), 'This file already exists.'
        torch.save(result.cpu(), f'{filename}.pt')

def save_array_(result, filename, i=0, add_duplicates=False):
    if add_duplicates:
        v_format = f'_v{i}' if i > 0 else ''
        if os.path.isfile(f'{filename}{v_format}.npy'):
            save_array_(result, f'{filename}', i=i+1, add_duplicates=True)
        else:
            if i == 0:
                np.save(f'{filename}.npy', result)
            else:
                print(f'Warning: Saved as version {i}')
                np.save(f'{filename}_v{i}.npy', result)
    else:
        assert not(os.path.isfile(f'{filename}.npy')), 'This file already exists.'
        np.save(f'{filename}.npy', result)

def save_tensors(results, filenames, add_duplicates=False):
    if isinstance(results, list):
        for result, filename in zip(results, filenames):
            save_tensor_(result, filename, 0, add_duplicates)
    else:
        save_tensor_(results, filenames, 0, add_duplicates)

def load_tensors(filenames):
    if isinstance(filenames, list):
        results = []
        for filename in filenames:
            results.append(torch.load(f'{filename}.pt', weights_only=True))
        return tuple(results)
    else:
        return torch.load(f'{filenames}.pt', weights_only=True)

def save_arrays(results, filenames, add_duplicates=False):
    if isinstance(results, list):
        for result, filename in zip(results, filenames):
            save_array_(result, filename, 0, add_duplicates)
    else:
        save_array_(results, filenames, 0, add_duplicates)

def load_arrays(filenames):
    if isinstance(filenames, list):
        results = []
        for filename in filenames:
            results.append(np.load(f'{filename}.npy'))
        return tuple(results)
    else:
        return np.load(f'{filenames}.npy')

def clear_files(filenames):
    if isinstance(filenames, list):
        for filename in filenames:
            assert os.path.isfile(filename), 'This file does not exist.'
            os.remove(filename)
