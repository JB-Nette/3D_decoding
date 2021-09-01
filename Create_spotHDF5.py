import h5py
import pandas as pd
import numpy as np
import os

data_path = 'C:/Users/Nette/Desktop/3D_gaussian/Testfullpipline_3D/'
coords_file_path = "C:/Users/Nette/Desktop/FOV_00_coord_iter1.hdf5"
smFish_fpkm_file = os.path.join(data_path,"FPKM_file.tsv")
df_fpkm = pd.read_csv(smFish_fpkm_file, header=None, sep="\t", usecols=[0, 1], names=['genes','FPKM'])
spots_coor = {}
spots_coor["gene1"]= np.array([[1,200,300],[2,400,500]]) #[z,y,x]
def CreateH5py(df_fpkm, coords_file_path, spots_coor):
    with h5py.File(coords_file_path, 'w') as coords_file:
        for i, genes in enumerate(spots_coor.keys()):
            print(genes)
            print(i)
            xy_coor = spots_coor[genes][0:3]
            list_of_spot_params = xy_coor
            print(list_of_spot_params)
        if list_of_spot_params is not None:
            print("x")
            gene_spots_data = list_of_spot_params
            gene_dataset = coords_file.create_dataset(
                genes, data=gene_spots_data
            )
        else:  # no spots found
            gene_dataset = coords_file.create_dataset(
                genes, shape=(0, 7)
            )
        dataset_attrs = {
            "gene_index": i,
            "FPKM_data": df_fpkm.FPKM[i],
        }

        for attr in dataset_attrs:
            gene_dataset.attrs[attr] = dataset_attrs[attr]

CreateH5py(df_fpkm, coords_file_path, spots_coor)