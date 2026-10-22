import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def set_style():
    """Applies clean, research-grade aesthetics for publication-quality figures."""
    plt.style.use('default')
    plt.rcParams.update({
        # Typography
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'axes.labelcolor': '#1e293b',
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.titlepad': 10,
        'axes.titlecolor': '#0f172a',
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9,
        # Spines and framing
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': '#94a3b8',
        'axes.linewidth': 1.0,
        # Gridlines (subtle and behind data)
        'axes.grid': True,
        'grid.color': '#e2e8f0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.8,
        'axes.axisbelow': True,
        # Lines & Markers
        'lines.linewidth': 2.2,
        'lines.markersize': 5.5,
        'lines.markeredgewidth': 1.2,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4.5,
        'ytick.major.size': 4.5,
        'xtick.color': '#475569',
        'ytick.color': '#475569',
        # Legend styling
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': '#ffffff',
        'legend.edgecolor': '#cbd5e1',
        'legend.fancybox': True,
        # Output quality
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
    })

def plot_label_distribution(dist_matrix, title, out_path):
    set_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    im = ax.imshow(dist_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('Samples per class', fontsize=10, labelpad=8, weight='bold', color='#1e293b')
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor('#cbd5e1')
    
    ax.set_xlabel('Class Label (0 to 9)')
    ax.set_ylabel('Client Index (0 to 99)')
    ax.set_title(title)
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(10))
    
    # Keep border for image visualization
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cbd5e1')
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_centralized_curves(csv_path, out_path):
    set_style()
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    
    palette = {'2NN': '#2563eb', 'CNN': '#059669'}
    markers = {'2NN': 'o', 'CNN': 's'}
    
    for model_name, group in df.groupby('model'):
        color = palette.get(model_name, '#2563eb')
        marker = markers.get(model_name, 'o')
        peak_acc = group['test_acc'].max() * 100
        label = f"{model_name} (Peak: {peak_acc:.1f}%)"
        ax.plot(group['epoch'], group['test_acc'] * 100, label=label,
                color=color, marker=marker, markersize=5.5, markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=1.4, lw=2.2)
        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Centralized Baseline Ceiling on MNIST')
    ax.set_ylim(85, 100)
    ax.legend(loc='lower right', title='Model Architecture')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_fedsgd_participation(csv_path, out_path):
    set_style()
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    
    c_colors = {
        0.0: '#94a3b8',
        0.1: '#38bdf8',
        0.2: '#0284c7',
        0.5: '#2563eb',
        1.0: '#1e1b4b',
    }
    c_styles = {
        0.0: ':',
        0.1: '--',
        0.2: '-.',
        0.5: '-',
        1.0: '-',
    }
    
    for c_val in sorted(df['C'].unique()):
        group = df[df['C'] == c_val]
        grouped = group.groupby('round')['test_acc'].agg(['mean', 'std']).reset_index()
        color = c_colors.get(c_val, '#2563eb')
        style = c_styles.get(c_val, '-')
        label = f'$C = {c_val}$ (1 client/round)' if c_val == 0.0 else f'$C = {c_val}$ ({int(c_val*100)} clients/round)'
        ax.plot(grouped['round'], grouped['mean'] * 100, label=label,
                color=color, linestyle=style, lw=2.2,
                marker='o' if c_val in [0.0, 1.0] else None, markersize=4.5,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.3)
        if grouped['std'].notna().any() and (grouped['std'] > 0).any():
            ax.fill_between(grouped['round'], (grouped['mean'] - grouped['std']) * 100,
                            (grouped['mean'] + grouped['std']) * 100, color=color, alpha=0.15)
            
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('FedSGD Client Participation ($C$ Fraction)')
    ax.set_ylim(0, 75)
    ax.legend(loc='upper left', title='Participation Rate')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_fedavg_computation(csv_path, out_path):
    set_style()
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    
    configs = [
        ((1, 600), '$E=1, B=\\mathrm{all}$ (1 step/round)', '#64748b', ':', '^'),
        ((1, 50), '$E=1, B=50$ (12 steps/round)', '#0284c7', '--', 's'),
        ((5, 10), '$E=5, B=10$ (300 steps/round)', '#16a34a', '-', 'o'),
    ]
    
    for (e_val, b_val), label, color, style, marker in configs:
        group = df[(df['E'] == e_val) & (df['B'] == b_val)]
        if group.empty:
            continue
        grouped = group.groupby('round')['test_acc'].agg(['mean', 'std']).reset_index()
        ax.plot(grouped['round'], grouped['mean'] * 100, label=label,
                color=color, linestyle=style, marker=marker, markersize=5.5,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5, lw=2.2)
        if grouped['std'].notna().any() and (grouped['std'] > 0).any():
            ax.fill_between(grouped['round'], (grouped['mean'] - grouped['std']) * 100,
                            (grouped['mean'] + grouped['std']) * 100, color=color, alpha=0.15)
            
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('FedAvg: Speedup from Local Computation ($E$ Epochs, $B$ Batch Size)')
    ax.set_ylim(40, 100)
    ax.legend(loc='lower right', title='Local Compute Budget')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_noniid_comparison(csv_path, out_path):
    set_style()
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    
    # Exact colors aligning with documentation
    part_order = [
        ('IID', 'IID (Uniform across 100 clients)', '#16a34a', '-', 'o'),
        ('Dirichlet 0.5', 'Dirichlet $\\alpha = 0.5$ (Mild label skew)', '#d97706', '-', 's'),
        ('Dirichlet 0.1', 'Dirichlet $\\alpha = 0.1$ (Severe label skew)', '#2563eb', '-', '^'),
        ('Pathological', 'Pathological 2-shard (Disjoint classes)', '#dc2626', '-', 'D'),
    ]
    
    for key, label, color, style, marker in part_order:
        group = df[df['partition'] == key]
        if group.empty:
            continue
        grouped = group.groupby('round')['test_acc'].agg(['mean', 'std']).reset_index()
        ax.plot(grouped['round'], grouped['mean'] * 100, label=label,
                color=color, linestyle=style, marker=marker, markersize=5.5,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5, lw=2.2)
        if grouped['std'].notna().any() and (grouped['std'] > 0).any():
            ax.fill_between(grouped['round'], (grouped['mean'] - grouped['std']) * 100,
                            (grouped['mean'] + grouped['std']) * 100, color=color, alpha=0.15)
            
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('FedAvg Convergence: The Non-IID Penalty')
    ax.set_ylim(10, 100)
    ax.legend(loc='lower right', title='Data Distribution')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_interpolation(csv_path, out_path):
    set_style()
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    
    modes = [
        ('Shared', 'Shared Initialization ($w_0$ shared)', '#d97706', '-'),
        ('Independent', 'Independent Initialization ($w_0$ random)', '#2563eb', '--'),
    ]
    
    for prefix, label, color, style in modes:
        group = df[df['mode'].str.lower().str.startswith(prefix.lower())]
        if not group.empty:
            ax.plot(group['theta'], group['loss'], label=label, color=color, linestyle=style, lw=2.4)
        
    ax.axvline(0.0, color='#94a3b8', linestyle=':', lw=1.1, alpha=0.9)
    ax.axvline(1.0, color='#94a3b8', linestyle=':', lw=1.1, alpha=0.9)
    ax.axvline(0.5, color='#cbd5e1', linestyle='--', lw=1.1, alpha=0.9)
    
    y_max = ax.get_ylim()[1]
    y_min = ax.get_ylim()[0]
    ax.text(0.0, y_max * 0.93, r' $w_B$', color='#475569', fontsize=10, ha='left', weight='bold')
    ax.text(1.0, y_max * 0.93, r'$w_A$ ', color='#475569', fontsize=10, ha='right', weight='bold')
    ax.text(0.5, y_min + (y_max - y_min) * 0.05, r'Midpoint $\theta=0.5$', color='#64748b', fontsize=8.5, ha='center')
    
    ax.set_xlabel(r'Interpolation Weight $\theta$ ($w(\theta) = \theta w_A + (1-\theta)w_B$)')
    ax.set_ylabel('Evaluation Loss on Full Dataset')
    ax.set_title('Weight Space Geometry: Loss Basin Connectivity')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_survey_curves(csv_path, out_path, x_col='round', y_col='test_acc', group_col='setting',
                       title='Survey Demo', x_label='Round', y_label='Test accuracy (%)'):
    set_style()
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    
    # Custom color and marker map for known survey experiment settings
    style_map = {
        # Settings
        'Cross-silo (K=10, C=1.0)': ('#059669', '-', 'o'),
        'Cross-device (K=100, C=0.1)': ('#2563eb', '--', 's'),
        # Non-IID Types
        'Label skew': ('#2563eb', '-', 'o'),
        'Quantity skew': ('#059669', '-', 's'),
        'Feature skew': ('#dc2626', '-', '^'),
        # Dropout
        'p = 0.0': ('#059669', '-', 'o'),
        'p = 0.2': ('#2563eb', '--', 's'),
        'p = 0.5': ('#d97706', ':', '^'),
        # Client Drift
        'IID': ('#2563eb', '-', 'o'),
        'Pathological Non-IID': ('#dc2626', '-', 's'),
    }
    
    fallback_colors = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#0d9488']
    fallback_markers = ['o', 's', '^', 'D', 'v', 'p']
    
    y_mult = 100 if (y_col == 'test_acc' and df[y_col].max() <= 1.0) else 1
    
    for idx, (val, group) in enumerate(df.groupby(group_col)):
        grouped = group.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        val_str = str(val)
        
        if val_str in style_map:
            color, linestyle, marker = style_map[val_str]
        else:
            color = fallback_colors[idx % len(fallback_colors)]
            linestyle = '-'
            marker = fallback_markers[idx % len(fallback_markers)]
            
        ax.plot(grouped[x_col], grouped['mean'] * y_mult, label=val_str,
                color=color, linestyle=linestyle, marker=marker, markersize=5.0,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.4, lw=2.2)
                
        if grouped['std'].notna().any() and (grouped['std'] > 0).any():
            ax.fill_between(grouped[x_col], (grouped['mean'] - grouped['std']) * y_mult,
                            (grouped['mean'] + grouped['std']) * y_mult, color=color, alpha=0.15)
            
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Legend placement
    loc = 'upper right' if 'Drift' in title else ('lower right' if 'accuracy' in y_label.lower() else 'best')
    ax.legend(loc=loc)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
