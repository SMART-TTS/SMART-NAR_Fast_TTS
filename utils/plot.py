import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrogram = spectrogram.numpy()[::-1]
    im = ax.imshow(spectrogram, aspect="auto")
    plt.colorbar(im, ax=ax)
    plt.xlabel("frames")
    plt.ylabel("fbank coeff")
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()
    return fig


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.T, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)

    # ylabel = 'Decoder timestep'
    # if info is not None:
    #     ylabel += '\n\n' + info
    # plt.ylabel(ylabel)
    # plt.xlabel('Encoder timestep')

    ylabel = 'Encoder timestep'
    if info is not None:
        ylabel += '\n\n' + info
    plt.ylabel(ylabel)
    plt.xlabel('Decoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()
    return fig


def plot_spectrum(spectrum, name, colorbar=False, dir='feats'):
    if not os.path.exists(dir):
        os.mkdir(dir)

    fig, ax = plt.subplots()
    # im = ax.imshow(np.flip(spectrum, 0), cmap="jet", aspect=0.2 * spectrum.shape[1] / spectrum.shape[0])
    # spectrum = spectrum.detach().numpy()[::-1]
    spectrum = np.flip(spectrum, 0)
    im = ax.imshow(spectrum, aspect="auto")

    if colorbar:
        fig.colorbar(im)
    plt.savefig('{}/{}.png'.format(dir, name), format='png')
    plt.close(fig)


def plot_attention(attention, name, colorbar=False, dir='feats'):
    """Plots the alignment.

    Args:
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      gs: (int) global step.
      dir: Output path.
    """
    if not os.path.exists(dir):
        os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(attention)
    if colorbar:
        fig.colorbar(im)
    plt.savefig('{}/{}.png'.format(dir, name), format='png')
    plt.close(fig)