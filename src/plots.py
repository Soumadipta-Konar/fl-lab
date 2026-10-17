import os
import pandas as pd
import matplotlib.pyplot as plt

def set_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.8
    })

def plot_label_distribution(dist_matrix, title, out_path):
    set_style()
    plt.figure(figsize=(8, 5))
    plt.imshow(dist_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Samples per class')
    plt.xlabel('Class label')
    plt.ylabel('Client index')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_centralized_curves(csv_path, out_path):
    set_style()
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 4.5))
    for model_name, group in df.groupby('model'):
        plt.plot(group['epoch'], group['test_acc'] * 100, label=model_name, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Test accuracy (%)')
    plt.title('Centralized Baseline Training on MNIST')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_fedsgd_participation(csv_path, out_path):
    set_style()
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 4.5))
    for c_val, group in df.groupby('C'):
        grouped = group.groupby('round')['test_acc'].agg(['mean', 'std']).reset_index()
        plt.plot(grouped['round'], grouped['mean'] * 100, label=f'C = {c_val}')
        if grouped['std'].notna().any():
            plt.fill_between(grouped['round'], (grouped['mean'] - grouped['std']) * 100,
                             (grouped['mean'] + grouped['std']) * 100, alpha=0.15)
    plt.xlabel('Communication round')
    plt.ylabel('Test accuracy (%)')
    plt.title('FedSGD Client Participation (C fraction)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_fedavg_computation(csv_path, out_path):
    set_style()
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 4.5))
    for (e_val, b_val), group in df.groupby(['E', 'B']):
        grouped = group.groupby('round')['test_acc'].agg(['mean', 'std']).reset_index()
        b_label = 'all' if b_val == 0 or b_val == 600 else f'{b_val}'
        plt.plot(grouped['round'], grouped['mean'] * 100, label=f'E={e_val}, B={b_label}')
        if grouped['std'].notna().any():
            plt.fill_between(grouped['round'], (grouped['mean'] - grouped['std']) * 100,
                             (grouped['mean'] + grouped['std']) * 100, alpha=0.15)
    plt.xlabel('Communication round')
    plt.ylabel('Test accuracy (%)')
    plt.title('FedAvg Effect of Local Computation (E epochs, B batch size)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_noniid_comparison(csv_path, out_path):
    set_style()
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 4.5))
    for part, group in df.groupby('partition'):
        grouped = group.groupby('round')['test_acc'].agg(['mean', 'std']).reset_index()
        plt.plot(grouped['round'], grouped['mean'] * 100, label=f'{part}')
        if grouped['std'].notna().any():
            plt.fill_between(grouped['round'], (grouped['mean'] - grouped['std']) * 100,
                             (grouped['mean'] + grouped['std']) * 100, alpha=0.15)
    plt.xlabel('Communication round')
    plt.ylabel('Test accuracy (%)')
    plt.title('FedAvg on IID vs Non-IID Dirichlet and Shards')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_interpolation(csv_path, out_path):
    set_style()
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 4.5))
    for mode, group in df.groupby('mode'):
        plt.plot(group['theta'], group['loss'], label=mode)
    plt.xlabel(r'$\theta$')
    plt.ylabel('Training loss')
    plt.title('Model Weight Interpolation: ' + r'$\theta w_A + (1-\theta) w_B$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_survey_curves(csv_path, out_path, x_col='round', y_col='test_acc', group_col='setting',
                       title='Survey Demo', x_label='Round', y_label='Test accuracy (%)'):
    set_style()
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 4.5))
    for val, group in df.groupby(group_col):
        grouped = group.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        y_mult = 100 if y_col == 'test_acc' else 1
        plt.plot(grouped[x_col], grouped['mean'] * y_mult, label=str(val))
        if grouped['std'].notna().any():
            plt.fill_between(grouped[x_col], (grouped['mean'] - grouped['std']) * y_mult,
                             (grouped['mean'] + grouped['std']) * y_mult, alpha=0.15)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
