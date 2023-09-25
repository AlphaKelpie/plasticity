import pandas as pd
# from subprocess import run, PIPE
from tqdm import tqdm
from pytorch_fid import fid_score

tipo = []
quan = []
rumo = []
neur = []
matr = []
epoc = []
outp = []
tota = []
rela = []

for face in tqdm(range(0, 5), desc='face', leave=True):
    for neurons in tqdm([5, 9, 13, 17, 21], desc='neur', leave=False):
        fid_tota = fid_score.calculate_fid_given_paths(['stat/total.npz',
                                                   f'curti/diff_10/{face}_{neurons}'],
                                            batch_size=64, device='cpu',
                                            dims=64, num_workers=0)
        
        # fid_rela = fid_score.calculate_fid_given_paths([f'stat/{face}.npz',
        #                                            f'5_same_face/{face}_{neurons}'],
        #                                     batch_size=64, device='cpu',
        #                                     dims=64, num_workers=0)
        
        tipo.append('diversi')
        quan.append(5)
        rumo.append(False)
        neur.append(neurons)
        matr.append('curti')
        epoc.append(3000)
        outp.append(64)
        tota.append(fid_tota)
        rela.append(-100.0)

# ind = list(range(0, len(tipo)))

df = pd.DataFrame(list(zip(tipo, quan, rumo, neur, matr, epoc, outp,
                           tota, rela)),
                  columns=['Tipo', 'Quantita', 'Rumore', 'Neuroni',
                           'Matrice', 'Epoche', 'Output', 'FID_Totale',
                           'FID_Relativo'])


data = pd.read_csv(filepath_or_buffer='olivetti_data.csv', sep=',', header=0, index_col=0)
data = pd.concat([data, df], ignore_index=True)

data.to_csv('olivetti_data.csv', index=True, index_label='Index')
