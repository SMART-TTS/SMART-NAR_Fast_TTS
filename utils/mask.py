import numpy as np

def get_guide_and_mask_for_attention(text_lengths, mel_lengths, guide_g=0.2, mask_g=0.15, r=None, c=None):
    b = len(text_lengths)
    if r is None:
        r = np.max(text_lengths)
    if c is None:
        c = np.max(mel_lengths)
    guide = np.ones((b, r, c), dtype=np.float32)
    mask = np.ones((b, r, c), dtype=np.float32)
    for i in range(b):
        W = guide[i]
        M = mask[i]
        N = float(text_lengths[i])
        T = float(mel_lengths[i])
        for n in range(r):
            for t in range(c):
                if n < N and t < T:
                    w = 1.0 - np.exp(-(float(n) / N - float(t) / T) ** 2 / (2.0 * (guide_g ** 2)))
                    W[n][t] = w
                    m = np.exp(-(float(n) / N - float(t) / T) ** 2 / (2.0 * (mask_g ** 2)))
                    M[n][t] = m
                elif t >= T and n < N:
                    w = 1.0 - np.exp(-((float(n - N - 1) / N) ** 2 / (2.0 * (guide_g ** 2))))
                    W[n][t] = w
                    m = np.exp(-((float(n - N - 1) / N) ** 2 / (2.0 * (mask_g ** 2))))
                    M[n][t] = m
    return guide, mask
