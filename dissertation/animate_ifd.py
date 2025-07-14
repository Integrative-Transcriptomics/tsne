import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import jax

from tueplots import cycler, fonts, fontsizes, bundles
from tueplots.constants import markers
from tueplots.constants.color import palettes
from sklearn.preprocessing import LabelEncoder
from matplotlib import colors

def animate(samples, labels, output, cmap):
    #plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
    #plt.rcParams.update(fonts.aistats2022_tex(family="serif"))
    #plt.rcParams.update(fontsizes.aistats2022())
    plt.rcParams['axes.grid'] = False

    le = LabelEncoder()
    cs = le.fit_transform(labels)
    labels_set = list(dict.fromkeys(labels))
    cmap = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=0, vmax=len(labels_set)-1)
    color_mapping = {label: cmap(norm(i)) for i, label in enumerate(labels_set)}

    fig, ax = plt.subplots(figsize=(7, 5))
    sample_0 = samples[:, 0]
    sample_0 = sample_0.reshape((len(labels), 2))
    minimum_x = np.min(np.array([i.reshape((len(labels), 2)) for i in samples.T])[:, :, 0])
    maximum_x = np.max(np.array([i.reshape((len(labels), 2)) for i in samples.T])[:, :, 0])
    minimum_y = np.min(np.array([i.reshape((len(labels), 2)) for i in samples.T])[:, :, 1])
    maximum_y = np.max(np.array([i.reshape((len(labels), 2)) for i in samples.T])[:, :, 1])
    ax.set_xlim((minimum_x - (0.1 * (maximum_x - minimum_x)), maximum_x + (0.1 * (maximum_x - minimum_x))))
    ax.set_ylim((minimum_y - (0.1 * (maximum_y - minimum_y)), maximum_y + (0.1 * (maximum_y - minimum_y))))
    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')
    scat = ax.scatter(sample_0[:, 0], sample_0[:, 1], c=[color_mapping[label] for label in labels])
    # Manually create a legend
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[label], markersize=8, label=label)
                  for label in labels_set]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.04, 1), loc="upper left")    
    plt.tight_layout()

    def init():
        return scat,

    def animate(i):
        sample_i = samples[:, i]
        sample_i = sample_i.reshape((len(labels), 2))
        scat.set_offsets(sample_i)
        return scat, 
        
    anim = FuncAnimation(
        fig, animate, interval=1000, frames=samples.shape[1], blit=True, init_func=init)
 
    #anim.save("ifd_results/ifd.mov", dpi=150, writer=FFMpegWriter(fps=5))
    #anim.save("laplace_approximation/laplace.mov", dpi=150, writer=FFMpegWriter(fps=5))
    anim.save(output, dpi=150, writer=PillowWriter(fps=5))
    #anim.save("laplace_approximation/laplace.gif", dpi=150, writer=PillowWriter(fps=1000))


def draw_and_plot_samples(mean, Y_unflattener, cov, n, outfile, cmap, key=42):
    key = jax.random.PRNGKey(key)
    samples = jax.random.multivariate_normal(key, mean, cov, shape=(n,))
    print(samples.shape)
    f, ax = plt.subplots(1)
    for j in samples:
        T = Y_unflattener(j)
        ax.scatter(T[:, 0], T[:, 1], c = [i for i in range(T.shape[0])], cmap=cmap)
        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')
    plt.savefig(outfile)
