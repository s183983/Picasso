import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np
import json
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from picasso.augmentor import Augment
import picasso.mesh.utils as meshUtil


class MeshDataset(Dataset):
    def __init__(self, file_list, root, lm_ids = 0, rendered_data=False, transform=None):
        self.file_list = file_list
        self.lab_dir = os.path.join(root,"labels")
        self.transform = transform
        self.lm_ids = lm_ids
        self.rendered_data = rendered_data


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file)
        reader.Update()
        vertices = np.array(reader.GetOutput().GetPoints().GetData())
        poly = np.array(dsa.WrapDataObject(reader.GetOutput()).Polygons)
        faces = np.reshape(poly,(-1,4))[:,1:4]
        
        lab_name = os.path.join(self.lab_dir,'_'.join(os.path.basename(file).split('_')[0:2])+".npy")
        loaded = np.load(lab_name)
        label = loaded[self.lm_ids].T


        args = [torch.from_numpy(vertices), torch.from_numpy(faces), torch.from_numpy(label)]

        if self.rendered_data:
            load_texture

        if self.transform:
            args = self.transform(args)

        # plain args:  vertex, face, nv, mf, label
        # render args: vertex, face, nv, mf, face_texture, bary_coeff, num_texture, label
        args.insert(2, torch.tensor([vertices.shape[0]]))
        args.insert(3, torch.tensor([faces.shape[0]]))
        return args


class default_collate_fn:
    def __init__(self, max_num_vertices):
        self.max_num_vertices = max_num_vertices

    def __call__(self, batch_data):
        batch_size = len(batch_data)
        batch_num_vertices = 0
        trunc_batch_size = 1
        for b in range(batch_size):
            batch_num_vertices += batch_data[b][2][0]
            if batch_num_vertices<self.max_num_vertices:
                trunc_batch_size = b+1
            else:
                break
        if trunc_batch_size<batch_size:
            print("The batch data is truncated.")

        batch_data = batch_data[:trunc_batch_size]
        batch_data = list(map(list, zip(*batch_data))) # transpose batch dimension and args dimension
        args_num = len(batch_data)
        batch_concat = []
        for k in range(args_num):
            batch_concat.append(torch.cat(batch_data[k], dim=0))
        return batch_concat








