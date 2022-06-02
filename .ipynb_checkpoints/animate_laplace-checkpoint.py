import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(5, 3))
    samples = np.load('laplace_approximation/samples.npy')
    labels = np.load('laplace_approximation/labels.npy')
    mean = np.load('laplace_approximation/mean.npy')
    print(labels)
    sample_0 = samples[:, 0]
    sample_0 = sample_0.reshape((len(labels), 2))
    minimum = np.min(samples)
    maximum = np.max(samples)
    ax.set_xlim((minimum, maximum))
    ax.set_ylim((minimum, maximum))
    scat = ax.scatter(sample_0[:, 0], sample_0[:, 1], c=labels, cmap='tab10')
    #scat, = ax.plot([], [], 'o')
    def init():
        #scat.set_data([], [])
        #plt.scatter(mean[:, 0], mean[:, 1], c='black')
        return scat,

    def animate(i):
        sample_i = samples[:, i]
        sample_i = sample_i.reshape((len(labels), 2))
        #scat.set_data(sample_i[:, 0], sample_i[:, 1])
        scat.set_offsets(sample_i)
        return scat, 
        
    anim = FuncAnimation(
        fig, animate, interval=1000, frames=samples.shape[1], blit=True, init_func=init)
 
    #anim.save("ifd_results/ifd.mov", dpi=150, writer=FFMpegWriter(fps=5))
    #anim.save("laplace_approximation/laplace.mov", dpi=150, writer=FFMpegWriter(fps=5))
    #anim.save("ifd_results/ifd.gif", dpi=150, writer=PillowWriter(fps=5))
    anim.save("laplace_approximation/laplace.gif", dpi=150, writer=PillowWriter(fps=5))