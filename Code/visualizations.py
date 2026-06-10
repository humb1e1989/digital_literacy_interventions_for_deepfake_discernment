import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

df_participants = pd.read_csv('../Data/all_participant_metrics.zip', compression='zip', header=0)
df_participants_follow_up = pd.read_csv('../Data/all_participant_metrics_followup.zip', compression='zip', header=0)
df_time = pd.read_csv('../Data/processed/pilot2/metrics_results_over_time.csv', header=0)

print(f"Main study participants: {df_participants.shape[0]}")
print(f"Follow-up participants:  {df_participants_follow_up.shape[0]}")
print(f"Longitudinal matched:   {df_time.shape[0]}")
print()
print("Fake accuracy by group (main study):")
print(df_participants.groupby('intervention_group')['fake_accuracy'].mean().round(2))
print()
print("Real accuracy by group (main study):")
print(df_participants.groupby('intervention_group')['real_accuracy'].mean().round(2))

order = ['control', 'intervention_2', 'intervention_3', 'intervention_4', 'intervention_5', 'intervention_1']
names = ['Control', 'Textual', 'Visual', 'Gamified', 'Feedback', 'Knowledge']
colors = ['#B22222', '#5C2D91', '#D16DB3', '#72AEE6', '#1F4B82', '#4D8E8A']


def capped_error(mean, std, n, cap=100):
    std_error = std / np.sqrt(n)
    lower = std_error
    upper = min(std_error, cap - mean)
    return lower, upper


def add_significance_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
    ax.text((x1 + x2) * .5, y + h - 1.2, text, ha='center', va='bottom', fontsize=20)


def plot_single_timepoint(df, file_name, metric, time_label, brackets=None, y_max=109, metric_label=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    spacing = 0.15
    positions = np.arange(len(order)) * spacing
    means = []

    for i, group in enumerate(order):
        group_data = df[df['intervention_group'] == group][metric]

        kde = gaussian_kde(group_data)
        x_range = np.linspace(group_data.min(), group_data.max(), 100)
        kde_values = kde(x_range)
        scaling_factor = 0.07
        kde_values = kde_values / kde_values.max() * scaling_factor

        ax.fill_betweenx(x_range, positions[i], positions[i] - kde_values,
                         facecolor=colors[i], alpha=0.7, edgecolor='black')

        mean_val = np.mean(group_data)
        std_val = np.std(group_data)
        n_val = len(group_data)
        lower, upper = capped_error(mean_val, std_val, n_val)
        ax.errorbar(positions[i] + 0.03, mean_val, yerr=[[lower], [upper]],
                    fmt='o', color=colors[i], capsize=5, markersize=7, elinewidth=2, capthick=2)
        means.append(mean_val)

    ax.plot([positions[0], positions[-1]], [means[0], means[0]], '--', color='black', alpha=0.3, linewidth=2)

    if brackets:
        for group1_idx, group2_idx, y_pos, height, text in brackets:
            add_significance_bracket(ax, positions[group1_idx], positions[group2_idx], y_pos, height, text)

    ax.set_xlim(positions[0] - 0.1, positions[-1] + 0.1)
    ax.set_ylim(0, y_max)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=12, rotation=45, ha='right')

    if metric_label is None:
        metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
        y_label = f"{metric_name} (in %)"
    else:
        y_label = metric_label

    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(False)

    yticks = ax.get_yticks()
    x_min, x_max = positions[0] - 0.1, positions[-1] + 0.1
    for y in yticks:
        ax.hlines(y, xmin=x_min, xmax=x_max, linestyle='--', alpha=0.7, color='lightgray', linewidth=1, zorder=0)

    ax.tick_params(axis='y', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(file_name, format='pdf', bbox_inches='tight', dpi=300)
    png_name = file_name.replace('.pdf', '.png')
    plt.savefig(png_name, format='png', bbox_inches='tight', dpi=150)
    print(f"Saved: {file_name}")
    plt.close()


# --- Real accuracy plots ---
plot_single_timepoint(df_participants,
                      file_name='../Results/pilot2/acc_real_t1.pdf',
                      metric='real_accuracy', time_label='T1',
                      metric_label='Real Image Accuracy (in %)')

plot_single_timepoint(df_participants_follow_up,
                      file_name='../Results/pilot2/acc_real_t2.pdf',
                      metric='real_accuracy', time_label='T2',
                      metric_label='Real Image Accuracy (in %)')

# --- Fake accuracy plots (with significance brackets from paper) ---
t1_brackets = [
    (0, 1, 100, 1.0, '**'),   # Control vs Textual
    (0, 2, 104, 1.0, '***'),  # Control vs Visual
]

plot_single_timepoint(df_participants,
                      file_name='../Results/pilot2/acc_fake_t1.pdf',
                      metric='fake_accuracy', time_label='T1',
                      metric_label='Deepfake Accuracy (in %)',
                      brackets=t1_brackets)

plot_single_timepoint(df_participants_follow_up,
                      file_name='../Results/pilot2/acc_fake_t2.pdf',
                      metric='fake_accuracy', time_label='T2',
                      metric_label='Deepfake Accuracy (in %)')

# --- Sharing intention plots ---
plot_single_timepoint(df_participants,
                      file_name='../Results/pilot2/sharing_fake_t1.pdf',
                      metric='fake_sharing_rate', time_label='T1',
                      metric_label='Fake Image Sharing Rate (in %)')

plot_single_timepoint(df_participants,
                      file_name='../Results/pilot2/sharing_real_t1.pdf',
                      metric='real_sharing_rate', time_label='T1',
                      metric_label='Real Image Sharing Rate (in %)')

print("\nAll plots saved to Results/pilot2/")
