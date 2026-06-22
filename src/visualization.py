import numpy              as np
import pandas             as pd
import seaborn            as sns

from   src.routines       import bootstrap_ci
from   scipy.stats        import gaussian_kde
from   skimage            import color, filters, measure
from   skimage.transform  import resize
from   skimage.morphology import skeletonize
from   scipy.stats        import gaussian_kde

from   matplotlib         import pyplot as plt

def print_histograms(real_features, synth_features):
    feature_names = list(real_features.keys())
    n = len(feature_names)

    fig, axes = plt.subplots(n, 1, figsize=(10, 5.2 * n), sharex=False)
    if n == 1:
        axes = np.atleast_1d(axes)

    for ax, feature_name in zip(axes, feature_names):
        real_data  = real_features[feature_name]
        synth_data = synth_features[feature_name]

        real_mean = np.nanmean(real_data)
        real_ci   = bootstrap_ci(real_data)

        synth_mean = np.nanmean(synth_data)
        synth_ci   = bootstrap_ci(synth_data)

        x_min = np.nanmin([np.nanmin(real_data), np.nanmin(synth_data)])
        x_max = np.nanmax([np.nanmax(real_data), np.nanmax(synth_data)])
        shared_range = (x_min, x_max)

        ax.hist(real_data,
                bins=30,
                alpha=0.6,
                color='#9999CC',
                label=f'Real (n={len(real_data)})',
                density=True,
                range=shared_range,
                edgecolor='black',
                linewidth=0.5)

        ax.hist(synth_data,
                bins=30,
                alpha=0.6,
                color='#FF9966',
                label=f'Synthetic (n={len(synth_data)})',
                density=True,
                range=shared_range,
                edgecolor='black',
                linewidth=0.5)

        x_vals = np.linspace(x_min, x_max, 300)

        real_clean  = np.asarray(real_data, dtype=float)
        synth_clean = np.asarray(synth_data, dtype=float)
        real_clean  = real_clean[~np.isnan(real_clean)]
        synth_clean = synth_clean[~np.isnan(synth_clean)]

        if len(np.unique(real_clean)) > 1:
            ax.plot(x_vals, gaussian_kde(real_clean)(x_vals),
                    color='#9999CC', linewidth=2)

        if len(np.unique(synth_clean)) > 1:
            ax.plot(x_vals, gaussian_kde(synth_clean)(x_vals),
                    color='#FF9966', linewidth=2)

        ax.axvline(real_mean, color='#9999CC', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(synth_mean, color='#FF9966', linestyle='--', linewidth=2, alpha=0.8)

        ax.axvspan(real_ci[0], real_ci[1], alpha=0.1, color='#9999CC', label='Real 95% CI')
        ax.axvspan(synth_ci[0], synth_ci[1], alpha=0.1, color='#FF9966', label='Synthetic 95% CI')

        title = f'Real vs. Synthetic {feature_name.replace("_", " ").title()} Distribution'
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(feature_name.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_xlim(shared_range)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Real vs Synthetic Feature Distributions', fontsize=15, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig


def visualize_gabor_and_skeleton_examples(image, mask, n_examples=4):
    """
    Visualizes Gabor filter responses and skeletons for a sample of nuclei.

    Parameters
    ----------
    image    : np.ndarray  - grayscale or RGB image
    mask     : np.ndarray  - integer label mask (each nucleus has a unique label)
    n_examples : int       - number of nuclei to visualize
    """

    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    if gray.shape != mask.shape:
        mask = resize(
            mask, gray.shape,
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(mask.dtype)

    gabor_frequencies  = [0.1, 0.3, 0.5]
    gabor_orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    gabor_responses = {}
    for freq in gabor_frequencies:
        for theta in gabor_orientations:
            real, imag = filters.gabor(gray, frequency=freq, theta=theta)
            gabor_responses[(freq, theta)] = np.sqrt(real**2 + imag**2)

    props = measure.regionprops(mask, intensity_image=gray)
    props = [p for p in props if p.image.shape[0] > 5 and p.image.shape[1] > 5]
    sampled = props[:n_examples]

    for i, prop in enumerate(sampled):
        minr, minc, maxr, maxc = prop.bbox
        nucleus_region = gray[minr:maxr, minc:maxc]
        nucleus_mask   = prop.image
        skeleton       = skeletonize(nucleus_mask)

        display_region = nucleus_region.copy()
        display_region[~nucleus_mask] = 0

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        fig.suptitle(f"Nucleus {i+1} — Skeleton", fontsize=13)

        axes[0].imshow(display_region, cmap='gray')
        axes[0].set_title("Grayscale crop")
        axes[0].axis('off')

        axes[1].imshow(nucleus_mask, cmap='gray')
        axes[1].set_title("Binary mask")
        axes[1].axis('off')

        axes[2].imshow(nucleus_mask, cmap='gray', alpha=0.5)
        axes[2].imshow(skeleton, cmap='hot', alpha=0.8)
        axes[2].set_title(f"Skeleton (length={int(np.sum(skeleton))})")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, len(gabor_frequencies) + 1, figsize=(14, 3))
        fig.suptitle(f"Nucleus {i+1} — Gabor Responses (mean over orientations)", fontsize=13)

        axes[0].imshow(display_region, cmap='gray')
        axes[0].set_title("Grayscale crop")
        axes[0].axis('off')

        for j, freq in enumerate(gabor_frequencies):
            stacked = np.stack([
                gabor_responses[(freq, theta)][minr:maxr, minc:maxc]
                for theta in gabor_orientations
            ], axis=0)
            mean_response = np.mean(stacked, axis=0)
            mean_response[~nucleus_mask] = 0

            axes[j+1].imshow(mean_response, cmap='inferno')
            axes[j+1].set_title(f"freq={freq}")
            axes[j+1].axis('off')

        plt.tight_layout()
        plt.show()

def create_coverage_barplot(coverage_df: pd.DataFrame):
    metrics = ['Variance', 'Entropy', 'Distance_to_Centroid', 'Convex_Hull_Volume']
    df_plot = coverage_df.reset_index().rename(columns={'index': 'Dataset'})
    df_melt = df_plot.melt(id_vars='Dataset', var_name='Metric', value_name='Value')

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(9, 4.8 * n))

    for ax, metric in zip(axes, metrics):
        data = df_melt[df_melt['Metric'] == metric]
        
        sns.barplot(
            data=data,
            x='Dataset',
            y='Value',
            ax=ax,
            palette='Set2',
            edgecolor='black',
            linewidth=0.7
        )
        
        ax.set_title(f'{metric}', fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel('')
        ax.set_ylabel('Value', fontsize=11)
        ax.tick_params(axis='x', rotation=0)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3, fontsize=9)

    fig.suptitle('Real vs Synthetic', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def create_congruence_barplot(congruence_df: pd.DataFrame):
    metrics = ['JSD', 'EMD_Wasserstein', 'Cosine_Similarity']
    
    df_melt = congruence_df.melt(
        id_vars=['Feature'],
        value_vars=metrics,
        var_name='Metric',
        value_name='Value'
    )

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))

    for ax, metric in zip(axes, metrics):
        data = df_melt[df_melt['Metric'] == metric].sort_values('Value', ascending=True)
        
        sns.barplot(
            data=data,
            y='Feature',
            x='Value',
            ax=ax,
            palette='viridis',
            edgecolor='black',
            linewidth=0.4
        )
        
        ax.set_title(metric, fontsize=20, fontweight='bold', pad=10)
        ax.set_ylabel('')
        ax.set_xlabel('Value', fontsize=11)
        ax.tick_params(axis='y', labelsize=8)

    fig.suptitle('Real vs. Synthetic',
                 fontsize=26, fontweight='bold', y=0.95)

    fig.subplots_adjust(top=0.75)

    return fig

# Add these two functions at the very end of visualization.py

def create_completeness_barplot(comp_df: pd.DataFrame):
    """Simple barplot for completeness metrics."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    metrics = ['Missing_Data_Percentage', 'Required_Fields_Completeness']
    df_melt = comp_df.melt(id_vars='Dataset', value_vars=metrics,
                           var_name='Metric', value_name='Value')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, metrics):
        data = df_melt[df_melt['Metric'] == metric]
        sns.barplot(data=data, x='Dataset', y='Value', ax=ax, palette='Set2', edgecolor='black')
        ax.set_title(metric.replace('_', ' '), fontsize=13, fontweight='bold')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

    fig.suptitle('Completeness: Real vs Synthetic', fontsize=15, fontweight='bold', y=0.975)
    fig.tight_layout()
    return fig


def create_consistency_barplot(cons_df: pd.DataFrame, group_by="hospital"):
    """Barplot for consistency metrics including ANOVA F-statistic."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    if cons_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No consistency data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    metrics_to_plot = ['Variance_of_Group_Means', 'Max_Min_Difference', 'ANOVA_F_statistic']
    
    df_melt = cons_df.melt(
        id_vars=['Metric'], 
        value_vars=metrics_to_plot,
        var_name='Consistency_Metric', 
        value_name='Value'
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, cmetric in zip(axes, metrics_to_plot):
        data = df_melt[df_melt['Consistency_Metric'] == cmetric]
        
        sns.barplot(
            data=data, 
            x='Metric', 
            y='Value', 
            ax=ax, 
            palette='coolwarm', 
            edgecolor='black'
        )
        
        ax.set_title(cmetric.replace('_', ' '), fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel('Value', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=2, fontsize=9)

    fig.suptitle(f'Consistency across {group_by}', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout()
    return fig