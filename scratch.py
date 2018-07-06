import numpy as np
import matplotlib.pyplot as plt

def viewCurrentVolume(preds):
    fig, ax = plt.subplots(2, 5)
    ax = list(ax[0]) + list(ax[-1])
    for idx, i in enumerate(range(0, 150, 15)):
        ax[idx].imshow(preds[i,0,:,:], cmap='gray')
    plt.show()

pred = np.load(open('./predictions[0].npz', 'rb'))

viewCurrentVolume(pred)

print('Hello')