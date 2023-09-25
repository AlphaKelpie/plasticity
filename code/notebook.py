import numpy as np
import pylab as plt
# import pandas as pd
# import seaborn as sns
# import tensorflow as tf
# from IPython.display import clear_output
from sklearn.datasets import fetch_olivetti_faces
from subprocess import run
from tqdm import tqdm

from plasticity.model import BCM
from plasticity.model.optimizer import Adam
from plasticity.model.weights import GlorotUniform, GlorotNormal
from plasticity.utils import view_weights

X, y = fetch_olivetti_faces(return_X_y=True)


# #   PRINT ALL OLIVETTI'S FACES
# for idx in range(20, 400):
#     img = X[idx].copy()
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#     ax.imshow(img.reshape(64, 64), cmap='gray')
#     ax.axis('off')
#     fig.savefig(f'olivetti/{idx}.png', transparent=True, bbox_inches='tight')
#     plt.close(fig)


# SELECT PARAMETERS
np.random.shuffle(X)
# inputs = X[:10].copy()
# n = inputs.shape[0]
# neurons = 16
# path = 'path'

batch = 10
epoch = 3*10**3

for face in tqdm(range(4, 5), leave=True, desc='face'):
    inputs = X[face*20 : (face+1)*20].copy()
    # n = inputs.shape[0]
    for neurons in tqdm([5, 9, 13, 17, 21], leave=False, desc='neur'):
        path = f"random20/{face}_{neurons}"

        run(['mkdir', '-p', path])
        for rn in tqdm(range(0, 64), leave=False, desc='rand'):
            model = BCM(outputs=neurons, num_epochs=epoch, batch_size=batch, interaction_strength=[-0.2, 0.3],
                optimizer=Adam(lr=1e-3), activation='leaky', weights_init=GlorotNormal(), verbose=False, random_state=rn)
            model.fit(inputs)
            view_weights(model.weights[neurons//2], dims=(64,64), cmap='gray', figsize=(5,5), path=f'{path}/{rn}.png')

# fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_ylabel('Sample')
# ax.set_xlabel('Neuron')
# t = model.predict(inputs).T
# ax.imshow(t)
# # plt.show()
