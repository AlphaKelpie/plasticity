import numpy as np
import pandas as pd
from cv2 import imread, IMREAD_GRAYSCALE
from prdc import compute_prdc
from sklearn.datasets import fetch_olivetti_faces
from tqdm import tqdm
from pytorch_fid import fid_score
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

fid = []
prec = []
rec = []
dens = []
cov = []

tipo = []
quan = []
rumo = []
neur = []
matr = []
epoc = []
outp = []
tota = []
rela = []

NEAREST_K = 5
BATCH = 10
FACES = list(range(0, 5))
NEURONS = [5, 9, 13, 17,21]
X, y = fetch_olivetti_faces(return_X_y=True)
arrays = [np.reshape(imread(f'olivetti/{num}.png', IMREAD_GRAYSCALE), -1)
            for num in range(400)]
real_features = np.stack(arrays.copy())
arrays[:] = []
for face in tqdm(FACES, leave=True, desc='faces'):    
    for neurons in tqdm(NEURONS, leave=False, desc='neur'):
        arrays = [np.reshape(imread(f'random20/{face}_{neurons}/{num}.png',
                                  IMREAD_GRAYSCALE), -1) for num in range(64)]
        fake_features = np.stack(arrays.copy())
        arrays[:] = []
        blockPrint()
        metrics = compute_prdc(real_features=real_features,
                             fake_features=fake_features,
                             nearest_k=NEAREST_K)
        enablePrint()
        prec.append(metrics['precision'])
        rec.append(metrics['recall'])
        dens.append(metrics['density'])
        cov.append(metrics['coverage'])

for face in tqdm(FACES, desc='faces', leave=True):
    for neurons in tqdm(NEURONS, desc='neur', leave=False):
        fid_tota = fid_score.calculate_fid_given_paths(['stat/total.npz',
                                                   f'random20/{face}_{neurons}'],
                                            batch_size=64, device='cpu',
                                            dims=64, num_workers=0)

            # fid_rela = fid_score.calculate_fid_given_paths([f'stat/{face}.npz',
            #                                            f'5_same_face/{face}_{neurons}'],
            #                                     batch_size=64, device='cpu',
            #                                     dims=64, num_workers=0)

        tipo.append('tutti')
        quan.append("random20")
        rumo.append(False)
        neur.append(neurons)
        matr.append('greg')
        epoc.append(3000)
        outp.append(64)
        tota.append(fid_tota)
        rela.append(-100.0)

df = pd.DataFrame(list(zip(tipo, quan, rumo, neur, matr, epoc, outp,
                           tota, rela, prec, rec, dens, cov)),
                  columns=['Tipo', 'Batch', 'Rumore', 'Neuroni',
                           'Matrice', 'Epoche', 'Output', 'FID_Totale',
                           'FID_Relativo', 'Precisione', 'Richiamo',
                           'Densita', 'Copertura']
                )
df.to_csv('olivetti_new1.csv', index=True, index_label='Index')
            # with open("prdc.txt", "a", encoding='utf-8') as myfile:
            #     myfile.write(dumps(metrics))

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
