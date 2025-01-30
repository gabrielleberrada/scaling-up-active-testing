import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import torch
import pandas as pd
import seaborn as sns
import os
import utils
from sklearn import metrics as sk_metrics
import seaborn.objects as so
import metrics
from scipy.stats import spearmanr

NAMES = {'llama2_7b': '7B-zero',
         'llama2_7b_icl10': '7B-few',
         'llama2_7b_icl20': '7B-few',
         'llama2_7b_icl30': '7B-few',
         'llama2_7b_icl40': '7B-few',
         'llama2_7b_icl50': '7B-few',
         'llama2_7b_icl50_v1': '7B-few',
         'llama2_70b': '70B-zero',
         'llama2_70b_icl10': '70B-few',
         'llama2_70b_icl20': '70B-few',
         'llama2_70b_icl30': '70B-few',
         'llama2_70b_icl40': '70B-few',
         'llama2_70b_icl50': '70B-few',
         'sst2': 'SST-2',
         'fpb': 'FPB',
         'agnews': 'AGN',
         'hatespeech': 'HS',
         'subj': 'Subj',
         'iid': 'Uniform random',
         'exp1': 'Experiment 1',
         'exp2': 'Experiment 2'
        }

NAMES_SURROGATE = {
         'llama2_7b_icl50': 'LURE-7B',
         'llama2_7b_icl40': 'LURE-7B',
         'llama2_70b_icl50': 'LURE-70B',
         'llama2_70b_icl40': 'LURE-70B'
        }


def setup_matplotlib(
    small_size: int = 4,
    medium_size: int = 5,
    bigger_size: int = 6,
):
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['lines.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['errorbar.capsize'] = 3
    plt.rcParams['figure.dpi'] = 400
    plt.rc('font', size=small_size)               # controls default text sizes
    plt.rc('axes', titlesize=medium_size)          # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)         # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)         # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)         # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)         # legend fontsize
    plt.rc('legend', title_fontsize=medium_size)  # legend title fontsize
    plt.rc('figure', titlesize=bigger_size)       # fontsize of the figure title

def vis_temperature_scaling(temperature, final_loss, scores, targets, loss, x_min, x_max):
    X = np.linspace(x_min, x_max, 1_000)
    Y = [loss(scores, targets, temperature=x) for x in X]
    plt.plot(X, Y)
    plt.scatter(temperature, final_loss, color='red', label='Optimised value')
    plt.xlabel('Temperature')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss as a function of temperature on the validation dataset')

def plot_all_errors(preds,
                    target,
                    step,
                    title='',
                    savefig=(False, None),
                   e=('pi', 10)):
    data = pd.DataFrame({})
    
    for method_name, pred in preds.items():
        log_mse_loss = metrics.log_mse(pred, target)
        mse_loss = metrics.se(pred, target)
        runs = pred.shape[1]
        length = len(mse_loss)
        if step == 1:
            iterations = np.repeat(np.arange(1, length+1), runs)
        else:
            iterations = np.repeat(np.arange(1, length, step=step), runs)

        data_mse = pd.DataFrame({'Number of acquired labels': iterations,
                                 'Log Squared Error': log_mse_loss.flatten(), 
                                 'Squared Error': mse_loss.flatten(),
                                 'Method': method_name})
        data = pd.concat((data, data_mse), ignore_index=True)
        
    fig, axs = plt.subplots(1, 3, figsize=(3.5, 1.5))
    
    sns.lineplot(data, 
                ax=axs[0],
                x='Number of acquired labels', 
                y='Log Squared Error', 
                hue='Method', 
                linestyle='dashdot',
                errorbar=('se', 2),
                legend=True)
    axs[0].set_title('Mean Log Squared Error +/- 2 SE')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_ylabel('Error')

    sns.lineplot(data, 
                ax=axs[1],
                x='Number of acquired labels', 
                y='Squared Error', 
                hue='Method', 
                linestyle='dashdot',
                errorbar=('se', 2),
                legend=True)
    axs[1].set_yscale('log')
    axs[1].set_title('Mean Squared Error +/- 2 SE')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_ylabel('')
    
    sns.lineplot(data, 
                ax=axs[2],
                x='Number of acquired labels', 
                y='Squared Error', 
                hue='Method', 
                linestyle='dashdot',
                errorbar=e,
                estimator='median',
                legend=True)
    axs[2].set_yscale('log')
    axs[2].set_title(f'Median Squared Error - PI 10')
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].set_ylabel('')
    
    handles, labels = axs[0].get_legend_handles_labels()
    for ax in axs:
        ax.get_legend().remove()
    fig.legend(handles, labels, loc='upper center', ncols=5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(title)
    
    if savefig[0]:
        plt.savefig(savefig[1])

def plot_comparison_errors_small(step,
                            datasets,
                            model,
                            files,
                            set_name='full',
                            title='',
                            ratios=(1.65, 1.3),
                            savefig=(False, None)):
    rows = len(datasets)//2
    columns = len(datasets)//rows
    fig, axs = plt.subplots(rows, columns, figsize=(columns*ratios[0], rows*ratios[1]))
    length = 1
    for j, dataset in enumerate(datasets):
        row = j // 2
        col = j % 2
        target = utils.load_tensors(f'{dataset}/{model}/{set_name if dataset != 'fpb' else 'active'}_set_loss').numpy()
        data = pd.DataFrame({})
        for method_name, file in files.items():
            pred = utils.load_arrays(f'{dataset}/{model}/{model}_{file}')
            if len(pred.shape) < 2:
                pred = np.array([pred]*length)
            mse_loss = metrics.se(pred, target)
            runs = pred.shape[1]
            length = len(mse_loss)
            if rows > 1:
                ax = axs[row, col]
            else:
                ax = axs[j]
            if step == 1:
                iterations = np.repeat(np.arange(1, length+1), runs)
            else:
                iterations = np.repeat(np.arange(1, length, step=step), runs)    
            
            data_mse = pd.DataFrame({'Number of acquired labels': iterations, 
                                'Squared Error': mse_loss.flatten(),
                                'Method': method_name})
            data = pd.concat((data, data_mse), ignore_index=True)
        sns.lineplot(data, 
                ax=ax,
                x='Number of acquired labels', 
                y='Squared Error', 
                hue='Method', 
                linestyle='dashdot',
                errorbar=('pi', 10),
                estimator='median',
                legend=True)
        ax.set_yscale('log')
        ax.set_title(f'{NAMES[dataset]}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('')
    if rows > 1:
        for row in range(rows):
            axs[row, 0].set_ylabel(f'Risk-estimation error')
            handles, labels = axs[row, 0].get_legend_handles_labels()
            for col in range(columns):
                axs[row, col].get_legend().remove()
    else:
        axs[0].set_ylabel(f'Risk-estimation error')
        handles, labels = axs[0].get_legend_handles_labels()
        for ax in axs:
            ax.get_legend().remove()
    fig.legend(handles, labels, loc='upper center', ncols=3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])#0.93
    fig.suptitle(title)
    
    if savefig[0]:
        plt.savefig(savefig[1])


def plot_barplots(data,
                step,
                loss,
                thresholds=np.array([50, 100, 200, 500, 800]),
                title='',
                savefig=(False, None)):
    df = pd.DataFrame({})
    for method_name, results in data.items():
        pred = loss[0](results[0], results[2])
        baseline = loss[0](results[1], results[2])
        values = baseline[thresholds - 1]
        corresp_values = np.argmax(np.expand_dims(pred, axis=-1) <= values, axis=-2)
        corresp_values[~np.any(np.expand_dims(pred, axis=-1) <= values, axis=0)] = len(pred)
        corresp_values[1:] = corresp_values[1:] - corresp_values[:-1]
        corresp_values = corresp_values / thresholds[-1]
        new_df = pd.DataFrame({'Model': method_name, 
                               loss[1]: corresp_values, 
                               'Thresholds': thresholds})
        aggregated_results = new_df.groupby(['Model', 'Thresholds']).sum().reset_index()
        df = pd.concat([df, aggregated_results], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    df['Thresholds'] = df['Thresholds'].astype(str)
    p = (so.Plot(df, x='Model', color='Thresholds', y=loss[1])
         .add(so.Bar(), 
        so.Stack())
         .scale(color="mako")
         .label(title=title, x='')
        )
    p.on(ax).plot()
    for ix, a in enumerate(ax.patches):
        x_start = a.get_x()
        width = a.get_width()
        for threshold in thresholds:
            ax.plot([x_start, x_start+width], [threshold/thresholds[-1]]*2, '--', c='dimgrey')
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    ax.tick_params(axis='x', rotation=90)

def get_method_name(data_row, extended=True):
    if 'ASE' in data_row and data_row['ASE']:
        return ' ASE'
    if data_row['NLL']:
        return ' NLL' + (' unw.' if ('IS' in data_row and data_row['IS']) else "")
    if data_row['Entropy']:
        return ' PE' + (' unw.' if ('IS' in data_row and data_row['IS']) else "")
    return ' CE' + (' unw.' if ('IS' in data_row and data_row['IS']) else "")

def plot_relative_cost(data,
                    loss,
                    title='',
                    savefig=(False, None),
                    xlim=None,
                    smoothing=1,
                    reduced=False,
                    subset='active'):
    df = pd.DataFrame({})
    relative_cost = []
    N = 1
    add = '_reduced' if reduced else ''
    for data_row in data.iterrows():
        data_row = data_row[1]
        if data_row.loc['Dataset'] == 'agnews' and (
            data_row.loc['Model'].endswith('icl50') or data_row.loc['Surrogate'].endswith('icl50')
        ):
            data_row.loc['Model'] = 'icl40'.join(data_row.loc['Model'].split('icl50'))
            data_row.loc['Surrogate'] = 'icl40'.join(data_row.loc['Surrogate'].split('icl50'))
            data_row.loc['File'] = 'icl40'.join(data_row.loc['File'].split('icl50'))
        path = f"{data_row.loc['Dataset']}/{data_row.loc['Model']}/"
        if not(os.path.isfile(f"{data_row.loc['Dataset']}/{data_row.loc['Model']}/{data_row.loc['File']}{add}.npy")):
            continue
        if data_row.loc['Dataset'] == 'fpb':
            target_error = utils.load_tensors(f'{path}active_set_loss').numpy()
        else:
            target_error = utils.load_tensors(f'{path}{subset}_set_loss').numpy()
        iid_error = utils.load_arrays(f"{path}{data_row.loc['Model']}_iid_loss{add}")
        predicted_error = utils.load_arrays(f"{data_row.loc['Dataset']}/{data_row.loc['Model']}/{data_row.loc['File']}{add}")
        if predicted_error.ndim < 2:
            predicted_error = np.expand_dims(predicted_error, axis=1)
        pred = loss(predicted_error, target_error)
        baseline = loss(iid_error, target_error)
        limits = pred <= baseline[:, None]
        corresp_values = np.argmax(limits, axis=1) + 1
        corresp_values = corresp_values/ np.arange(1, len(baseline)+1)
        corresp_values = metrics.sma(corresp_values[np.any(limits, axis=1)], smoothing)[:len(pred)]
        df = pd.concat((df, pd.DataFrame({
                                'Model': np.repeat(NAMES[data_row['Model']], len(corresp_values)),
                                'Surrogate': np.repeat(NAMES_SURROGATE[data_row['Surrogate']] + 
                                                       get_method_name(data_row),
                                            len(corresp_values)),
                                'Dataset': np.repeat(NAMES[data_row['Dataset']], len(corresp_values)),
                                'Number of acquired labels': np.arange(len(corresp_values)),
                                'Relative Labelling Cost': corresp_values,
                                #'Error': np.repeat(0., len(corresp_values))
                            })))
        N = max(N, len(corresp_values))
    corresp_values = np.ones(N)
    for model in df['Model'].unique():
        df = pd.concat((df, pd.DataFrame({
                                    'Model': np.repeat(model, N),
                                    'Surrogate': np.repeat('I.I.D.', N),
                                    'Dataset': np.repeat('', N),
                                    'Number of acquired labels': np.arange(N),
                                    'Relative Labelling Cost': corresp_values,
                                    #'Error': np.repeat(0., N)
                                })))
    palette = sns.color_palette("tab10", n_colors=4)
    datasets = df['Dataset'].unique()
    y = df['Relative Labelling Cost'].max()
    if len(df.groupby('Model')) > 1:
        fig, axs = plt.subplots(1, len(df.groupby('Model')), figsize=(6, 2), layout="tight")
        for i, (model, df_) in enumerate(df.groupby('Model')):
            sns.lineplot(df_, style='Surrogate', y='Relative Labelling Cost', x='Number of acquired labels', 
                         hue='Dataset', palette=palette, ax=axs[i], style_order=df['Surrogate'].unique())
            axs[i].set_title(model)
            if xlim:
                axs[i].set_xlim(0, xlim)
                axs[i].set_ylim(-0.05, min(y, 2.))
            if i >0:
                axs[i].set_ylabel('')
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
        handles, labels = axs[-1].get_legend_handles_labels()
        for ax in axs:
            ax.get_legend().remove()
        splitting_index = labels.index('Surrogate')
        dataset_handles = handles[:splitting_index]
        surrogate_handles = handles[splitting_index:]
        dataset_labels = labels[:splitting_index]
        surrogate_labels = labels[splitting_index:]
        ncols = (len(dataset_labels)+1)//2
        fig.legend(dataset_handles, dataset_labels, loc="upper center", bbox_to_anchor=(0.27, 1.12),
                   ncols=ncols)
        ncols = (len(surrogate_labels)+1)//2
        fig.legend(surrogate_handles, surrogate_labels, loc="upper center", 
                   bbox_to_anchor=(0.82 if ncols == 3 else 0.77, 1.12), 
                   ncols=ncols)
    else:
        fig, axs = plt.subplots(1, len(df.groupby('Model')), figsize=(4, 2), layout="tight")
        sns.lineplot(df, style='Surrogate', y='Relative Labelling Cost', x='Number of acquired labels',
                     hue='Dataset', palette=palette, ax=axs)
        axs.set_title(df['Model'].unique()[0])
        if xlim:
            axs.set_xlim(0, xlim)
            axs.set_ylim(-0.05, min(y, 2.))
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        handles, labels = axs.get_legend_handles_labels()
        axs.get_legend().remove()
        plt.subplots_adjust(right=0.7)
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.3, 0.5), borderaxespad=0.)
    fig.suptitle(title)
    plt.tight_layout()
    if savefig[0]:
        fig.savefig(savefig[1], bbox_inches="tight")

def plot_relative_error(data,
                        loss,
                        title='',
                        savefig=(False, None),
                        xlim=None,
                        ylim=2,
                        smoothing=1,
                        reduced=False,
                        ase=False,
                        subset='active'):
    df = pd.DataFrame({})
    N = 1
    add = '_reduced' if reduced else ''
    for data_row in data.iterrows():
        data_row = data_row[1]
        if data_row.loc['Dataset'] == 'agnews' and (
            data_row.loc['Model'].endswith('icl50') or data_row.loc['Surrogate'].endswith('icl50')
        ):
            data_row.loc['Model'] = 'icl40'.join(data_row.loc['Model'].split('icl50'))
            data_row.loc['Surrogate'] = 'icl40'.join(data_row.loc['Surrogate'].split('icl50'))
            data_row.loc['File'] = 'icl40'.join(data_row.loc['File'].split('icl50'))
        path = f"{data_row.loc['Dataset']}/{data_row.loc['Model']}/"
        if not(os.path.isfile(f"{data_row.loc['Dataset']}/{data_row.loc['Model']}/{data_row.loc['File']}{add}.npy")):
            continue
        if data_row.loc['Dataset'] == 'fpb':
            target_error = utils.load_tensors(f'{path}active_set_loss').numpy()
        else:
            target_error = utils.load_tensors(f'{path}{subset}_set_loss').numpy()
        iid_error = utils.load_arrays(f"{path}{data_row.loc['Model']}_iid_loss{add}")
        predicted_error = utils.load_arrays(f"{data_row.loc['Dataset']}/{data_row.loc['Model']}/{data_row.loc['File']}{add}")
        if predicted_error.ndim < 2:
            predicted_error = np.expand_dims(predicted_error, axis=-1)
        pred = loss(predicted_error, target_error)[:xlim]
        baseline = loss(iid_error, target_error)[:xlim]
        relative_error = pred / baseline
        relative_error = metrics.sma(relative_error, smoothing)
        df = pd.concat((df, pd.DataFrame({
                                'Model': np.repeat(NAMES[data_row['Model']], len(relative_error)),
                                'Surrogate': np.repeat(NAMES[data_row['Surrogate']] + 
                                                       get_method_name(data_row),
                                            len(relative_error)),
                                'Dataset': np.repeat(NAMES[data_row['Dataset']], len(relative_error)),
                                'Number of acquired points': np.arange(len(relative_error)),
                                'Relative Error': relative_error,
                            })))
        N = max(N, len(relative_error))
        if ase:
            if os.path.isfile(f"{data_row.loc['Dataset']}/{data_row.loc['Model']}/{data_row.loc['Model']}_{data_row.loc['Surrogate']}_ase_loss{add}.npy"):
                ase_error = np.array([utils.load_arrays(f"{data_row.loc['Dataset']}/{data_row.loc['Model']}/{data_row.loc['Model']}_{data_row.loc['Surrogate']}_ase_loss{add}")]*N)
                pred_ase = loss(ase_error, target_error)[:xlim]
                relative_error_ase = metrics.sma(pred_ase / baseline, smoothing)
                df = pd.concat((df, pd.DataFrame({
                                    'Model': np.repeat(NAMES[data_row['Model']], N),
                                    'Surrogate': np.repeat(NAMES_SURROGATE[data_row['Surrogate']] + 
                                                        get_method_name(data_row) + " ASE",
                                                N),
                                    'Dataset': np.repeat(NAMES[data_row['Dataset']], N),
                                    'Number of acquired labels': np.arange(N),
                                    'Relative Error': relative_error_ase,
                                })))
    relative_error = np.ones(N)
    for model in df['Model'].unique():
        df = pd.concat((df, pd.DataFrame({
                                    'Model': np.repeat(model, N),
                                    'Surrogate': np.repeat('Uniform random', N),
                                    'Dataset': np.repeat('', N),
                                    'Number of acquired labels': np.arange(N),
                                    'Relative Error': relative_error,
                                })))
    palette = sns.color_palette("tab10", n_colors=4)
    y = df['Relative Error'].max()
    if len(df.groupby('Model')) > 1:
        fig, axs = plt.subplots(len(df.groupby('Model')), 1, figsize=(3, 2.5), layout="tight")
        for i, (model, df_) in enumerate(df.groupby('Model')):
            sns.lineplot(df_, style='Surrogate', y='Relative Error', x='Number of acquired labels', 
                         hue='Dataset', palette=palette, ax=axs[i], style_order=df['Surrogate'].unique())
            axs[i].set_title(model)
            if xlim:
                axs[i].set_xlim(0, xlim)
                axs[i].set_ylim(-0.05, min(y, ylim))
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
        handles, labels = axs[-1].get_legend_handles_labels()
        for ax in axs:
            ax.get_legend().remove()
        splitting_index = labels.index('Surrogate')
        dataset_handles = handles[:splitting_index]
        surrogate_handles = handles[splitting_index:]
        dataset_labels = labels[:splitting_index]
        surrogate_labels = labels[splitting_index:]
        ncols = (len(dataset_labels)+1)//2
        fig.legend(dataset_handles, dataset_labels, loc="upper center", bbox_to_anchor=(0.21, 1.05),
                   ncols=ncols)
        ncols = (len(surrogate_labels)+1)//2
        fig.legend(surrogate_handles, surrogate_labels, loc="upper center", 
                   bbox_to_anchor=(0.82 if ncols == 3 else 0.71, 1.05), 
                   ncols=ncols)
    else:
        fig, axs = plt.subplots(1, len(df.groupby('Model')), figsize=(3, 1.3), layout="tight")
        sns.lineplot(df, style='Surrogate', y='Relative Error', x='Number of acquired labels',
                     hue='Dataset', palette=palette, ax=axs)
        axs.set_title(df['Model'].unique()[0])
        if xlim:
            axs.set_xlim(0, xlim)
            axs.set_ylim(-0.05, min(y, 2.))
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        handles, labels = axs.get_legend_handles_labels()
        axs.get_legend().remove()
        plt.subplots_adjust(right=0.7)
        splitting_index = labels.index('Surrogate')
        dataset_handles = handles[:splitting_index]
        surrogate_handles = handles[splitting_index:]
        dataset_labels = labels[:splitting_index]
        surrogate_labels = labels[splitting_index:]
        ncols = (len(dataset_labels)+1)//2
        fig.legend(dataset_handles, dataset_labels, loc="upper center", bbox_to_anchor=(0.2, 1.12),
                   ncols=ncols)
        ncols = (len(surrogate_labels)+1)//2
        fig.legend(surrogate_handles, surrogate_labels, loc="upper center", 
                   bbox_to_anchor=(0.82 if ncols == 3 else 0.7, 1.12), 
                   ncols=ncols)
    fig.suptitle(title)
    plt.tight_layout()
    if savefig[0]:
        fig.savefig(savefig[1], bbox_inches="tight")

def plot_bootstrap_errors(datasets, model_file, surrogates, xlim=None, aspect=1., savefig=(False, ''), title=''):
    data = pd.DataFrame({})
    for dataset_file in datasets:
        for surrogate_file in surrogates:
            if dataset_file == 'agnews':
                model_file_ = '40'.join(model_file.split('50'))
                surrogate_file = '40'.join(surrogate_file.split('50'))
            else:
                model_file_ = model_file
            true_error = utils.load_tensors(f'{dataset_file}/{model_file_}/active_set_loss').numpy()
            preds = {
                'IID': utils.load_arrays(f'{dataset_file}/{model_file_}/{model_file_}_iid_loss')[:xlim],
                'IID Bootstrap': utils.load_arrays(f'{dataset_file}/{model_file_}/{model_file_}_iid_bootstrap_loss')[:xlim],
                'LLaMa2 70B ICL CE': utils.load_arrays(f'{dataset_file}/{model_file_}/{model_file_}_{surrogate_file}_loss')[:xlim],
                'LLaMa2 70B ICL CE Bootstrap': utils.load_arrays(f'{dataset_file}/{model_file_}/{model_file_}_{surrogate_file}_bootstrap_loss')[:xlim]
            }
            step = 1
            for method_name, pred in preds.items():
                runs = pred.shape[1]
                length = len(pred)
                if step == 1:
                    iterations = np.repeat(np.arange(1, length+1), runs)
                else:
                    iterations = np.repeat(np.arange(1, length, step=step), runs)
                df = pd.DataFrame({'N° Acquired Points': iterations, 
                                 'Loss prediction': metrics.se(pred, true_error).flatten(),
                                 'Method': method_name,
                                  'Dataset': NAMES[dataset_file],
                                  'Surrogate': NAMES_SURROGATE[surrogate_file]})
                data = pd.concat((data, df), ignore_index=True)
            iterations = np.arange(1, length+1) if step == 1 else np.arange(1, length, step=step)
    g = sns.relplot(data,
                x='Number of acquired labels', 
                y='Loss prediction', 
                hue='Method', 
                col='Dataset',
                row='Surrogate',
                kind='line',
                errorbar=('se', 2),
                legend=True,
                aspect=aspect)
    g.set(yscale='log')
    g.fig.suptitle(NAMES[model_file])
    g.fig.tight_layout(rect=[0, 0, 0.9, 0.98])
    if savefig[0]:
        plt.savefig(savefig[1])

def plot_bootstrap_error_estimation(data,
                                    title='',
                                    savefig=(False, None),
                                    xlim=None,
                                    ylim=None,
                                    smoothing=1,
                                    reduced=False,
                                    confidence_interval=2,
                                    subset='active'):
    df = pd.DataFrame({})
    add = '_v1' if reduced else ''
    for data_row in data.iterrows():
        data_row = data_row[1]
        if data_row.loc['Dataset'] == 'agnews' and (
            data_row.loc['Model'].endswith('icl50') or data_row.loc['Surrogate'].endswith('icl50')
        ):
            data_row.loc['Model'] = 'icl40'.join(data_row.loc['Model'].split('icl50'))
            data_row.loc['Surrogate'] = 'icl40'.join(data_row.loc['Surrogate'].split('icl50'))
            data_row.loc['File'] = 'icl40'.join(data_row.loc['File'].split('icl50'))
        path = f"{data_row.loc['Dataset']}/{data_row.loc['Model']}/"
        post = '_entropy' if data_row['Entropy'] else ('_nll' if data_row['NLL'] else '')
        if not(os.path.isfile(f"{path}/{data_row.loc['File']}_at_loss{post}{add}.npy")):
            continue
        if data_row.loc['Dataset'] == 'fpb':
            target_error = utils.load_tensors(f'{path}active_set_loss').numpy()
        else:
            target_error = utils.load_tensors(f'{path}{subset}_set_loss').numpy()
        single_run_error = utils.load_arrays(f"{path}/{data_row.loc['File']}_at_loss{post}{add}")
        bootstrap_variance = utils.load_arrays(f"{path}/{data_row.loc['File']}_bootstrap_variance{post}{add}")
        bootstrap_variance = np.sqrt(bootstrap_variance)
        in_interval = metrics.sma(np.logical_and((single_run_error + confidence_interval*bootstrap_variance) >= target_error,
                       (single_run_error-confidence_interval*bootstrap_variance) <= target_error).mean(axis=1), smoothing)
        df = pd.concat((df, pd.DataFrame({
                                'Model': np.repeat(NAMES[data_row['Model']], len(in_interval)),
                                'Surrogate': np.repeat(NAMES[data_row['Surrogate']] + 
                                                       get_method_name(data_row, extended=False),
                                            len(in_interval)),
                                'Dataset': np.repeat(NAMES[data_row['Dataset']], len(in_interval)),
                                'Number of acquired points': np.arange(len(in_interval)),
                                'Coverage probability': in_interval,
                            })))
    palette = sns.color_palette("tab10", n_colors=4)
    if len(df.groupby('Model')) > 1:
        fig, axs = plt.subplots(len(df.groupby('Model')), 1, figsize=(3, 2.5), layout="tight")
        for i, (model, df_) in enumerate(df.groupby('Model')):
            sns.lineplot(df_, style='Surrogate', y='Coverage probability', x='Number of acquired points', 
                         hue='Dataset', palette=palette,
                         ax=axs[i])
            axs[i].set_title(model)
            axs[i].set_ylim(-0.05, 1.)
            if xlim:
                axs[i].set_xlim(0, xlim)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].axhline(y=0.95, xmin=0, xmax=axs[i].get_xlim()[1], linestyle='--', color='dimgray')
        handles, labels = axs[-1].get_legend_handles_labels()
        for ax in axs:
            ax.get_legend().remove()
        splitting_index = labels.index('Surrogate')
        dataset_handles = handles[:splitting_index]
        surrogate_handles = handles[splitting_index:]
        dataset_labels = labels[:splitting_index]
        surrogate_labels = labels[splitting_index:]
        fig.legend(dataset_handles, dataset_labels, loc="upper center", bbox_to_anchor=(0.27, 1.05), ncols=3)
        fig.legend(surrogate_handles, surrogate_labels, loc="upper center", bbox_to_anchor=(0.77, 1.05), ncols=2)
    else:
        fig, axs = plt.subplots(1, 1, figsize=(3, 1.3), layout="tight")
        sns.lineplot(df, style='Surrogate', y='Coverage probability', x='Number of acquired points',
                     hue='Dataset', palette=palette, ax=axs, errorbar=None, 
                    )
        axs.set_title(f"""{df['Model'].unique()[0]}""")
        axs.set_ylim(-0.05, 1.)
        if xlim:
            axs.set_xlim(0, xlim)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.axhline(y=0.95, xmin=0, xmax=axs.get_xlim()[1], linestyle='--', color='dimgray')
        handles, labels = axs.get_legend_handles_labels()
        axs.get_legend().remove()
        ncols = (len(labels)+1)//2
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), borderaxespad=0., ncols=ncols)
    fig.suptitle(title)
    plt.tight_layout()
    if savefig[0]:
        fig.savefig(savefig[1], bbox_inches="tight")
    return df
        
    
def plot_relative_cost_selection(data,
                                loss,
                                title='',
                                savefig=(False, None),
                                xlim=None,
                                smoothing=1,
                                aspect=1.4,
                                reduced=False,
                                subset='active',
                                height=4.394):
    df = pd.DataFrame({})
    N = 1
    add = '_reduced' if reduced else ''
    for data_row in data.iterrows():
        data_row = data_row[1]
        if data_row.loc['Dataset'] == 'agnews' and data_row.loc['Surrogate'].endswith('icl50'):
            data_row.loc['Surrogate'] = 'icl40'.join(data_row.loc['Surrogate'].split('icl50'))
            data_row.loc['File'] = 'icl40'.join(data_row.loc['File'].split('icl50'))
        path = f"{data_row.loc['Dataset']}/{data_row.loc['Experiment']}/"
        if not(os.path.isfile(f"{data_row.loc['Dataset']}/{data_row.loc['Experiment']}/{data_row.loc['File']}{add}.npy")):
            continue
        if data_row.loc['Dataset'] == 'fpb':
            target_error = utils.load_tensors(f'{path}active_set_loss').numpy()
        else:
            target_error = utils.load_tensors(f'{path}{subset}_set_loss').numpy()
        iid_error = np.min(utils.load_arrays(f"{path}iid_loss{add}"), axis=2)
        predicted_error = np.min(utils.load_arrays(f"{path}{data_row.loc['File']}{add}"), axis=2)
        if predicted_error.ndim < 2:
            predicted_error = np.expand_dims(predicted_error, axis=1)
        pred = loss(predicted_error, target_error)
        baseline = loss(iid_error, target_error)
        limits = pred <= baseline[:, None]
        corresp_values = np.argmax(limits, axis=1) + 1
        corresp_values = corresp_values/ np.arange(1, len(baseline)+1)
        corresp_values = metrics.sma(corresp_values[np.any(limits, axis=1)], smoothing)
        df = pd.concat((df, pd.DataFrame({
                                'Experiment': np.repeat(NAMES[data_row['Experiment']], len(corresp_values)),
                                'Surrogate': np.repeat(NAMES[data_row['Surrogate']] + 
                                                       get_method_name(data_row),
                                            len(corresp_values)),
                                'Dataset': np.repeat(NAMES[data_row['Dataset']], len(corresp_values)),
                                'Number of acquired points': np.arange(len(corresp_values)),
                                'Relative Labelling Cost': corresp_values,
                            })))
        N = max(N, len(corresp_values))
    corresp_values = np.ones(N)
    for method in df['Method'].unique():
        df = pd.concat((df, pd.DataFrame({
                                    'Method': np.repeat(method, N),
                                    'Surrogate': np.repeat('I.I.D.', N),
                                    'Dataset': np.repeat('', N),
                                    'Number of acquired points': np.arange(N),
                                    'Relative Labelling Cost': corresp_values,
                                })))
    palette = sns.color_palette("tab10", n_colors=4)
    y = df['Relative Labelling Cost'].max()
    g = sns.relplot(df, style='Surrogate', y='Relative Labelling Cost', x='Number of acquired points', hue='Dataset',  kind='line', aspect=aspect, palette=palette, col='Experiment', errorbar=None, height=height)
    g.fig.suptitle(title)
    g.set(ylim=(-0.05, min(y, 2.25)))
    if xlim:
        g.set(xlim=(0, xlim))
    if savefig[0]:
        plt.savefig(savefig[1])


def plot_comparison_errors(step,
                            datasets,
                            models,
                            files,
                            set_name='full',
                            title='',
                            ratios=(1.65, 1.3),
                            savefig=(False, None)):
    rows = len(models)
    columns = len(datasets)
    fig, axs = plt.subplots(rows, columns, figsize=(columns*ratios[0], rows*ratios[1]))
    length = 1
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            target = utils.load_tensors(f'{dataset}/{model}/{set_name if dataset != 'fpb' else 'active'}_set_loss').numpy()
            data = pd.DataFrame({})
            for method_name, file in files.items():
                pred = utils.load_arrays(f'{dataset}/{model}/{model}_{file}')
                if len(pred.shape) < 2:
                    pred = np.array([pred]*length)
                mse_loss = metrics.se(pred, target)
                runs = pred.shape[1]
                length = len(mse_loss)
                if rows > 1 and columns > 1:
                    ax = axs[i, j]
                elif rows > 1:
                    ax = axs[i]
                else:
                    ax = axs[j]
                if step == 1:
                    iterations = np.repeat(np.arange(1, length+1), runs)
                else:
                    iterations = np.repeat(np.arange(1, length, step=step), runs)    
                
                data_mse = pd.DataFrame({'Number of acquired labels': iterations, 
                                    'Squared Error': mse_loss.flatten(),
                                    'Method': method_name})
                data = pd.concat((data, data_mse), ignore_index=True)
            sns.lineplot(data, 
                    ax=ax,
                    x='N° Acquired Points', 
                    y='Squared Error', 
                    hue='Method', 
                    linestyle='dashdot',
                    errorbar=('pi', 10),
                    estimator='median',
                    legend=True)
            ax.set_yscale('log')
            if i == 0:
                ax.set_title(f'{NAMES[dataset]}')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel('')
    if rows > 1:
        for row, model in enumerate(models):
            axs[row, 0].set_ylabel(f'{NAMES[model]}')
            handles, labels = axs[row, 0].get_legend_handles_labels()
            for col in range(columns):
                axs[row, col].get_legend().remove()
    else:
        axs[0].set_ylabel(f'MedSE +/- 10 %')
        handles, labels = axs[0].get_legend_handles_labels()
        for ax in axs:
            ax.get_legend().remove()
    fig.legend(handles, labels, loc='upper center', ncols=5)
    fig.tight_layout(rect=[0, 0, 0.99, 0.97])
    fig.suptitle(title)
    
    if savefig[0]:
        plt.savefig(savefig[1])

def plot_bootstrap_mse(data,
                    title='',
                    savefig=(False, None),
                    xlim=None,
                    smoothing=1,
                    confidence_interval=2,
                    reduced=False,
                    subset='active'):
    df = pd.DataFrame({})
    add = '_v1' if reduced else ''
    for data_row in data.iterrows():
        data_row = data_row[1]
        if data_row.loc['Dataset'] == 'agnews' and (
            data_row.loc['Model'].endswith('icl50') or data_row.loc['Surrogate'].endswith('icl50')
        ):
            data_row.loc['Model'] = 'icl40'.join(data_row.loc['Model'].split('icl50'))
            data_row.loc['Surrogate'] = 'icl40'.join(data_row.loc['Surrogate'].split('icl50'))
            data_row.loc['File'] = 'icl40'.join(data_row.loc['File'].split('icl50'))
        path = f"{data_row.loc['Dataset']}/{data_row.loc['Model']}/"
        post = '_entropy' if data_row['Entropy'] else ('_nll' if data_row['NLL'] else '')
        if not(os.path.isfile(f"{path}/{data_row.loc['File']}_at_loss{post}{add}.npy")):
            continue
        if data_row.loc['Dataset'] == 'fpb':
            target_error = utils.load_tensors(f'{path}active_set_loss').numpy()
        else:
            target_error = utils.load_tensors(f'{path}{subset}_set_loss').numpy()
        single_run_error = utils.load_arrays(f"{path}/{data_row.loc['File']}_at_loss{post}{add}")
        bootstrap_variance = utils.load_arrays(f"{path}/{data_row.loc['File']}_bootstrap_variance{post}{add}")
        true_error = metrics.se(single_run_error, target_error)
        mse = metrics.sma(np.median((bootstrap_variance-true_error)**2, axis=1), smoothing)
        in_interval = metrics.sma(np.logical_and((single_run_error + confidence_interval*np.sqrt(bootstrap_variance)) >= target_error,
                       (single_run_error-confidence_interval*np.sqrt(bootstrap_variance)) <= target_error).mean(axis=1), smoothing)
        df = pd.concat((df, pd.DataFrame({
                                'Target model': np.repeat(NAMES[data_row['Model']], len(mse)),
                                'Surrogate': np.repeat(NAMES_SURROGATE[data_row['Surrogate']],
                                            len(mse)),
                                'Dataset': np.repeat(NAMES[data_row['Dataset']], len(mse)),
                                'Number of acquired labels': np.arange(len(mse)),
                                'Median relative error': mse,
                                'Coverage probability': in_interval,
                            })))
    palette = sns.color_palette("tab10", n_colors=4)
    fig, axs = plt.subplots(2, len(df.groupby('Dataset')), figsize=(6.6, 2), layout="tight")
    for j, (dataset, df_) in enumerate(df.groupby(['Dataset'])):
            sns.lineplot(df_, style='Surrogate', y='Median relative error', x='Number of acquired labels',
                        hue='Target model', palette=palette,
                        ax=axs[0, j])
            axs[0, j].set_title(f'{dataset[0]}')
            axs[0, j].set_yscale('log')
            if xlim:
                axs[0, j].set_xlim(0, xlim)
            axs[0, j].set_ylabel('')
            axs[0, j].spines['top'].set_visible(False)
            axs[0, j].spines['right'].set_visible(False)
            axs[0, j].set_ylabel('MSE-estimation error')

            sns.lineplot(df_, style='Surrogate', y='Coverage probability', x='Number of acquired labels',
                    hue='Target model', palette=palette,
                    ax=axs[1, j])
            axs[1, j].set_title(f'{dataset[0]}')
            axs[1, j].set_ylim(-0.05, 1.)
            if xlim:
                axs[1, j].set_xlim(0, xlim)
            axs[1, j].spines['top'].set_visible(False)
            axs[1, j].spines['right'].set_visible(False)
            axs[1, j].axhline(y=0.95, xmin=0, xmax=axs[1, j].get_xlim()[1], linestyle='--', color='dimgray')
            axs[1, j].set_ylabel('Coverage probability')
    handles, labels = axs[-1, -1].get_legend_handles_labels()
    for ax in axs.flatten():
        ax.get_legend().remove()
    splitting_index = labels.index('Surrogate')
    model_handles = handles[:splitting_index]
    surrogate_handles = handles[splitting_index:]
    model_labels = labels[:splitting_index]
    surrogate_labels = labels[splitting_index:]
    fig.legend(model_handles, model_labels, loc="upper center", bbox_to_anchor=(0.35, 1.05), ncols=5)
    fig.legend(surrogate_handles, surrogate_labels, loc="upper center", bbox_to_anchor=(0.77, 1.05), ncols=3)
    fig.suptitle(title)
    plt.tight_layout()
    if savefig[0]:
        fig.savefig(savefig[1], bbox_inches="tight")