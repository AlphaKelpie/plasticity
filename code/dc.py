import numpy as np
from cv2 import imread, IMREAD_GRAYSCALE
from prdc import compute_prdc
from sklearn.datasets import fetch_olivetti_faces
from tqdm import tqdm
import sys, os
from json import dumps

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

ls1 = [5, 9, 13, 17, 21]
ls2 = [10, 20, 30, 40, 50]
NEAREST_K = 5
X, y = fetch_olivetti_faces(return_X_y=True)
arrays = [np.reshape(imread(f'olivetti/{num}.png', IMREAD_GRAYSCALE), -1)
          for num in range(400)]
real_features = np.stack(arrays.copy())
arrays[:] = []
for idx1 in tqdm(range(1, 9), leave=True, desc='type'):
    ls = []
    if idx1 in [4, 3]:
        ls = ls2
    else:
        ls = ls1
    for idx2 in tqdm(ls, leave=False, desc='neur'):
        for idx3 in tqdm(range(5), leave=False, desc='batc'):
            arrays = [np.reshape(imread(f'1/{idx1}/{idx3}_{idx2}/{num}.png',
                                        IMREAD_GRAYSCALE), -1) for num in range(64)]
            fake_features = np.stack(arrays.copy())
            arrays[:] = []
            blockPrint()
            metrics = compute_prdc(real_features=real_features,
                                   fake_features=fake_features,
                                   nearest_k=NEAREST_K)
            enablePrint()
            with open("prdc.txt", "a", encoding='utf-8') as myfile:
                myfile.write(dumps(metrics))

# import json
# import pandas as pd

# with open('prdc_.txt','r', encoding='utf-8') as file:
#     d = json.load(file)
# # print(type(d['foo'][0]))
# df = pd.DataFrame(d['foo'])
# data = pd.read_csv('olivetti_data.csv', sep=',', header=0, index_col=0)
# data1 = data.join(df)
# # print(data1.head())
# # print(data1.info())
# data1.to_csv('olivetti.csv', index=True, index_label='Index')
