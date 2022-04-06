import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import numpy as np
from math import log, floor, ceil
from torch_scatter import scatter_mean


from .modules.decimate import mesh_decimation_, compute_triangle_geometry_, \
                              count_vertex_adjface_, combine_clusters_


def mesh_decimation(vertexIn, faceIn, geometryIn, nvIn, mfIn, nvSample, useArea=True, wgtBnd=1.):
    vertexOut, faceOut, geometryOut, nvOut, mfOut, \
    repOut, mapOut = mesh_decimation_(vertexIn, faceIn, geometryIn, nvIn, mfIn,
                                      nvSample, useArea=useArea, wgtBnd=wgtBnd)
    return vertexOut, faceOut, geometryOut, nvOut, mfOut, repOut, mapOut


# triangle geometry include [unit length normals, intercept of the triangle plane, face area]
def compute_triangle_geometry(vertex, face):
    geometry = compute_triangle_geometry_(vertex, face)
    return geometry


# Count the number of adjacent faces for output vertices, i.e. vertices with vtMap[i]>=0
def count_vertex_adjface(face, vtMap, vertexOut):
    nf_count = count_vertex_adjface_(face.int(), vtMap.int(), vertexOut)
    return nf_count


def combine_clusters(rep_highRes, map_highRes, rep_lowRes, map_lowRes):
    rep_combined, map_combined = combine_clusters_(rep_highRes, map_highRes, rep_lowRes, map_lowRes)
    return rep_combined, map_combined


class MeshHierarchy:
    def __init__(self, *params):
        self.num_params = len(params)
        if self.num_params==3:
            self.nv_samples, self.useArea, self.wgtBnd = params
            self.Iters = len(self.nv_samples)
        elif self.num_params==4:
            self.stride, self.min_nvOut, self.useArea, self.wgtBnd = params
            self.Iters = len(self.stride)

    def __call__(self, vertex_in, face_in, nv_in, mf_in):
        mesh_hierarchy = []
        geometry_in = compute_triangle_geometry(vertex_in, face_in)
        mesh_hierarchy.append((vertex_in, face_in, geometry_in, nv_in, mf_in, None, None))
        for l in range(self.Iters):
            if self.num_params==3:
                nv_out = self.nv_samples[l]
            elif self.num_params==4:
                nv_out = (nv_in.to(torch.float)/float(self.stride[l])).to(torch.int)
                nv_out = torch.clip(nv_out, min=self.min_nvOut[l])
            dec_mesh = mesh_decimation(vertex_in, face_in, geometry_in, nv_in, mf_in,
                                       nv_out, self.useArea, self.wgtBnd)
            mesh_hierarchy.append(dec_mesh)
            vertex_in, face_in, geometry_in, nv_in, mf_in = dec_mesh[:5]
        return mesh_hierarchy


def crop_mesh(vertex, face, min_xyz, max_xyz, vertex_labels=None, facet_labels=None, crop_hgt=False):
    if crop_hgt:
        crop_vertex_mask = (vertex[:,0]>=min_xyz[0]) & (vertex[:,0]<=max_xyz[0]) &\
                           (vertex[:,1]>=min_xyz[1]) & (vertex[:,1]<=max_xyz[1]) &\
                           (vertex[:,2]>=min_xyz[2]) & (vertex[:,2]<=max_xyz[2])
    else:
        crop_vertex_mask = (vertex[:,0]>=min_xyz[0]) & (vertex[:,0]<=max_xyz[0]) &\
                           (vertex[:,1]>=min_xyz[1]) & (vertex[:,1]<=max_xyz[1])
    face = face.to(torch.long)  # create a copy of `face' of int64
    crop_face_mask = crop_vertex_mask[face[:,0],:] & \
                     crop_vertex_mask[face[:,1],:] & \
                     crop_vertex_mask[face[:,2],:]
    crop_face = face[crop_face_mask,:]
    crop_xyz_indices, idx = torch.unique(crop_face.view(-1),sorted=True,return_inverse=True)
    num_vertices = crop_xyz_indices.shape[0]

    if num_vertices==0:
        return None

    crop_face = idx.view([-1,3])
    crop_vertex = vertex[crop_xyz_indices,:]
    crop_labels = None
    if vertex_labels is not None:
        crop_labels = vertex_labels[crop_xyz_indices,:]
    if facet_labels is not None:
        crop_labels = facet_labels[crop_face_mask,:]
    crop_face = crop_face.to(torch.int)
    return crop_vertex, crop_face, crop_labels # tuple


def voxelize_mesh(vertex, face, voxel_size, seg_labels=None, return_inverse=False):
    assert (voxel_size>0, "")
    # print (colored("We assume the vertex size of per-mesh is less than 1 billion.","red"))

    labels = seg_labels.view(-1,1).to(torch.float)
    feature = torch.cat([vertex,labels],dim=1)

    min_xyz = vertex[:,:3].min(dim=0)[0]
    pos_xyz = vertex[:,:3] - min_xyz
    tmp_ids = torch.floor(pos_xyz/(voxel_size/100)) # centimeter voxel_size to meter
    Mags = torch.ceil(torch.log(torch.max(tmp_ids,dim=0)[0])/log(10.))

    ids = tmp_ids.to(torch.int)
    Mags = Mags.to(torch.int)

    ids_1D = ids[:,0]*(10**(Mags[1]+Mags[0])) + ids[:,1]*(10**Mags[0]) + ids[:,2]
    id_unique, idx = torch.unique(ids_1D,sorted=True,return_inverse=True)

    mean = scatter_mean(feature, idx, dim=0)
    new_vertex = mean[:,:-1]
    voxelized_labels = mean[:,-1:]

    # get labels, and face ids
    sorted_labels = voxelized_labels[idx,...]
    residual = torch.abs(sorted_labels - labels)
    edge_voxel_indices = torch.unique(idx[residual[:,0]>1e-4])
    voxelized_labels[edge_voxel_indices] = -1
    new_labels = voxelized_labels[:,0].to(torch.int)

    V1 = idx[face[:,0]]
    V2 = idx[face[:,1]]
    V3 = idx[face[:,2]]
    non_degenerate = ~((V1==V2) | (V2==V3) | (V3==V1))
    voxelized_face = torch.stack([V1,V2,V3],dim=1)
    new_face = voxelized_face[non_degenerate,:]

    if return_inverse:
        return new_vertex, new_face, new_labels, idx
    else:
        return new_vertex, new_face, new_labels

