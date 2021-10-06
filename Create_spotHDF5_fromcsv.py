import h5py
import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from numpy import genfromtxt


root = tk.Tk()
root.withdraw()
data_path = filedialog.askdirectory(title="Please select data directory")
root.destroy()

data_path = 'C:/Users/Nette/Desktop/3D_gaussian/Testfullpipline_3D/'
coords_file_path = data_path + 'FOV_00_coord_iter1.hdf5'
smFish_fpkm_file = os.path.join(data_path,"FPKM_file.tsv")
df_fpkm = pd.read_csv(smFish_fpkm_file, header=None, sep="\t", usecols=[0, 1], names=['genes','FPKM'])
spots_coor_path = os.path.join(data_path,'coor_csv')
spots_coor = {}
spots_coor_file_list = os.listdir(spots_coor_path)

for spots_csv in spots_coor_file_list:
    spots_csv_file = os.path.join(spots_coor_path,spots_csv)
    df_spots = pd.read_csv(spots_csv_file, header=None).to_numpy(dtype=int)
    df_spots = df_spots[:,1:4] #ignore first col
    df_spots = df_spots[1::] #ignore first row
    gene_name = spots_csv.split("_")[0]
    spots_coor[gene_name] = df_spots

    with h5py.File(coords_file_path, 'w') as coords_file:
        for i, genes in enumerate(spots_coor.keys()):
            print(genes)
            xy_coor = spots_coor[genes]
            list_of_spot_params = xy_coor
            print(list_of_spot_params)
            if list_of_spot_params is not None:
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

