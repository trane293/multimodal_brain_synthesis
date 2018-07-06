import numpy as np
import matplotlib.pyplot as plt
import glob as glob

def viewCurrentVolume(preds, title):
    fig, ax = plt.subplots(2, 5)
    ax = list(ax[0]) + list(ax[-1])
    for idx, i in enumerate(range(0, 150, 15)):
        ax[idx].imshow(preds[i,0,:,:], cmap='gray')
    plt.suptitle(title)
    plt.show()

t = len(glob.glob('./PREDICTIONS/*'))
for i in range(t):
    pred = np.load(open('./PREDICTIONS/pred_T1_T2-->T2FLAIR_[{}].npz'.format(i), 'rb'))

    viewCurrentVolume(pred, title='Prediction = {}'.format(i))

print('Hello')