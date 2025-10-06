# Import necessary libraries for plotting and data manipulation
import matplotlib as mpl
import matplotlib.pyplot as plt
from tueplots import bundles  # Provides publication-quality plotting styles from Tübingen University
from tueplots.constants.color import palettes # Access to specific color palettes for styling

# --- Global Plotting Configuration ---
# Increase the default resolution (dots per inch) of all generated plots for higher quality.
plt.rcParams.update({"figure.dpi": 150})
# Import libraries for creating animations
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
# Import a utility to encode categorical labels into numerical format
from sklearn.preprocessing import LabelEncoder

# Import core numerical and data handling libraries
import numpy as np
import gzip, pickle
from os import path
import seaborn as sns # For advanced and aesthetically pleasing statistical plots
from scipy import stats # For statistical functions, e.g., truncated normal distribution

# Define a custom diverging colormap. It transitions from a blueish color to white (center) to a reddish color.
# This is ideal for visualizing data that diverges from a central value (like zero).
cmap = mpl.colors.LinearSegmentedColormap.from_list("", [palettes.tue_plot[3], "white", palettes.tue_plot[0]])

class MidpointNormalize(mpl.colors.Normalize):
    """
    A custom matplotlib normalization class to center the colormap on a specific midpoint.

    This ensures that a given value (e.g., zero) maps exactly to the center of the
    colormap (e.g., white), making it easy to distinguish between positive and
    negative values in a heatmap.
    """
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

def equipotential_standard_normal(d, n):
    '''
    Draws `n` samples from a d-dimensional standard normal distribution that are
    "equipotential" (i.e., have the same probability density).

    These points lie on a great circle on a d-sphere, where the sphere's radius
    is determined by a random draw. This is a
    way to generate structured perturbations for sensitivity analysis.

    Args:
        d (int): The number of dimensions.
        n (int): The number of samples to generate.

    Returns:
        np.ndarray: An array of `n` samples of size `d` that are equally likely under a Gaussian.
    '''
    x = np.random.standard_normal((d, 1))  # starting sample
    
    r = np.sqrt(np.sum(x ** 2))  # ||x||
    x = x / r  # project sample on d-1-dimensional UNIT sphere --> x just defines direction
    t = np.random.standard_normal((d, 1))  # draw tangent sample
    t = t - (np.dot(np.transpose(t), x) * x)  # Gram Schmidth orthogonalization --> determines which circle is traversed
    t = t / (np.sqrt(np.sum(t ** 2)))  # standardize ||t|| = 1
    s = np.linspace(0, 2 * np.pi, n+1)  # space to span --> once around the circle in n steps
    s = s[0:(len(s) - 1)]
    t = s * t #if you wrap this samples around the circle you get once around the circle
    X = r * exp_map(x, t)  # project onto sphere, re-scale
    return (X)

def equipotential_standard_normal_within_one_std(d, n):
    """
    Similar to `equipotential_standard_normal`, but the initial random sample
    is drawn from a truncated standard normal distribution (values between -1 and 1).
    
    This results in a perturbation path that is closer to the origin.
    
    Args:
        d (int): The number of dimensions.
        n (int): The number of samples to generate.

    Returns:
        np.ndarray: An array of `n` samples of size `d`.
    """
    x = np.expand_dims(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=d), 1)    
    r = np.sqrt(np.sum(x ** 2))  # ||x||
    x = x / r  # project sample on d-1-dimensional UNIT sphere --> x just defines direction
    t = np.random.standard_normal((d, 1))  # draw tangent sample
    t = t - (np.dot(np.transpose(t), x) * x)  # Gram Schmidth orthogonalization --> determines which circle is traversed
    t = t / (np.sqrt(np.sum(t ** 2)))  # standardize ||t|| = 1
    s = np.linspace(0, 2 * np.pi, n+1)  # space to span --> once around the circle in n steps
    s = s[0:(len(s) - 1)]
    t = s * t #if you wrap this samples around the circle you get once around the circle
    X = r * exp_map(x, t)  # project onto sphere, re-scale
    return (X)

def exp_map(mu, E):
    '''
    The exponential map for a sphere. It maps a set of tangent vectors `E` at a
    point `mu` on the sphere back onto the sphere's surface.

    This is used to "walk" along the surface of the sphere.

    Args:
        mu (np.ndarray): The starting point on the unit sphere.
        E (np.ndarray): A matrix of tangent vectors at `mu`.

    Returns:
        np.ndarray: The resulting points on the unit sphere.
    '''
    D = np.shape(E)[0]
    theta = np.sqrt(np.sum(E ** 2, axis=0))
    np.seterr(invalid='ignore')
    M = np.dot(mu, np.expand_dims(np.cos(theta), axis=0)) + E * np.sin(theta) / theta
    if (any(np.abs(theta) <= 1e-7)):
        for a in (np.where(np.abs(theta) <= 1e-7)):
            M[:, a] = mu
    M[:, abs(theta) <= 1e-7] = mu
    return (M)

# --- Diagnostic Plotting Functions ---
def plot_heatmap(x, figsize=(5, 5), outfile=None, with_cell_lines=True):
    """
    Generates and optionally saves a heatmap with a diverging colormap centered at zero.

    Args:
        x (np.ndarray): The 2D data matrix to plot.
        figsize (tuple): The size of the figure.
        outfile (str, optional): If provided, the path to save the plot. If None, returns the axis object.
        with_cell_lines (bool): If True, adds thin grey lines to separate cells in the heatmap.

    Returns:
        matplotlib.axes.Axes or None: The axis object if outfile is None, otherwise None.
    """
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", [palettes.tue_plot[3], "white", palettes.tue_plot[0]])
    f, (ax1) = plt.subplots(1, 1, figsize=figsize)
    if with_cell_lines:
        sns.heatmap(x, cmap=cmap, norm=(MidpointNormalize(midpoint=0, vmin=np.min(x), vmax=np.max(x))), ax=ax1, linecolor='grey', linewidth=.5)
    else:
        sns.heatmap(x, cmap=cmap, norm=(MidpointNormalize(midpoint=0, vmin=np.min(x), vmax=np.max(x))), ax=ax1)
    
    if outfile==None:
        return ax1
    else:
        plt.savefig(outfile)

def animate(samples, labels, output, cmap):
    """
    Creates and saves an animation of t-SNE embeddings over time.

    Args:
        samples (np.ndarray): A numpy array of shape (n_points * 2, n_frames)
                              containing the sequence of 2D embeddings.
        labels (np.ndarray): The labels for each point, used for coloring.
        output (str): The path to save the output animation file (e.g., "animation.gif").
        cmap (list): A list of colors, e.g. in RGB format.
    """
    plt.rcParams['axes.grid'] = False

    # Encode labels to integers for color mapping
    le = LabelEncoder()
    color_encoder = le.fit_transform(labels)
    labels_dict = dict(zip(color_encoder, labels))
    cs = [cmap[i] for i in color_encoder]
    labels_set = list(dict.fromkeys(labels))

    # Determine the axis limits by finding the min/max coordinates across ALL frames
    # This prevents the plot from resizing during the animation.
    fig, ax = plt.subplots(figsize=(7, 5))
    sample_0 = samples[:, 0]
    sample_0 = sample_0.reshape((len(labels), 2))
    minimum_x = np.min(np.array([i.reshape((len(labels), 2)) for i in samples.T])[:, :, 0])
    maximum_x = np.max(np.array([i.reshape((len(labels), 2)) for i in samples.T])[:, :, 0])
    minimum_y = np.min(np.array([i.reshape((len(labels), 2)) for i in samples.T])[:, :, 1])
    maximum_y = np.max(np.array([i.reshape((len(labels), 2)) for i in samples.T])[:, :, 1])
    
    # Set axis limits with a 10% margin
    ax.set_xlim((minimum_x - (0.1 * (maximum_x - minimum_x)), maximum_x + (0.1 * (maximum_x - minimum_x))))
    ax.set_ylim((minimum_y - (0.1 * (maximum_y - minimum_y)), maximum_y + (0.1 * (maximum_y - minimum_y))))
    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')

    # Initialize the scatter plot with the first frame of data
    scat = ax.scatter(sample_0[:, 0], sample_0[:, 1], c=cs)

    # Manually create a legend
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap[label], markersize=8, label=labels_dict[label])
                  for label in labels_set]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.04, 1), loc="upper left")    
    plt.tight_layout()

    def init():
        """Initializes the animation. Returns the artist to be updated."""
        return scat,

    def animate(i):
        """
        The update function for each frame of the animation.
        
        Args:
            i (int): The current frame index.
        """
        sample_i = samples[:, i]
        sample_i = sample_i.reshape((len(labels), 2))
        scat.set_offsets(sample_i)
        return scat, 
        
    anim = FuncAnimation(
        fig, animate, interval=1000, frames=samples.shape[1], blit=True, init_func=init)
    
    # Save the animation as a GIF file
    # PillowWriter is used for creating GIFs. FFMpegWriter can be used for movie formats like .mp4 or .mov.
    anim.save(output, dpi=150, writer=PillowWriter(fps=5))