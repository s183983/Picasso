import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import picasso.mesh.utils as meshUtil
from picasso.mesh.layers import V2FConv3d, PerItemConv3d, BuildF2VCoeff
from picasso.mesh.layers import Pool3d as mesh_pool
from picasso.mesh.layers import Unpool3d as mesh_unpool
from .pi2_blocks import DualBlock, EncoderMeshBlock, FirstBlock


class PicassoNetII(nn.Module):
    def __init__(self, num_class, stride=None, mix_components=27, use_height=False):
        super(PicassoNetII,self).__init__()
        
        self.predFacet = False
        if stride is None:
            self.stride = [1.5, 1.5, 1.5, 1.5]
        else:
            assert(len(stride)==4)
            self.stride = stride
            
        self.device = "cuda"
        self.num_class = num_class
        self.num_clusters = mix_components
        self.useArea, self.wgtBnd = (True, 1.) # recommended hyper-parameters for mesh decimation
        self.temperature = 0.1
        self.bn_momentum = 0.01
        self.num_blocks = 5
        self.Niters = [2, 2, 4, 4, 4]
        self.radius = [0.2, 0.4, 0.8]
        self.EncodeChannels = [32, 64, 96, 128, 192, 256]
        self.useHeight = use_height
        if self.useHeight:
            geometry_in_channels = 12
        else:
            geometry_in_channels = 9

        self.min_nv = 80  # set the minimum vertex number for the coarsest layer
        self.stride = np.float32(self.stride)
        self.nv_limit = np.int32(np.cumprod(self.stride[::-1])[::-1]*self.min_nv)

        self.build_mesh_hierarchy = meshUtil.MeshHierarchy(self.stride, self.nv_limit,
                                                           self.useArea, self.wgtBnd)

        self.cluster = []
        for k in range(self.num_blocks):
            self.cluster.append(BuildF2VCoeff(self.num_clusters, tau=self.temperature))

        # FirstBlock
        self.Conv0 = FirstBlock(geometry_in_channels, self.EncodeChannels[0], self.num_clusters,
                                bn_momentum=self.bn_momentum)

        # setting of convolution blocks in the encoder
        self.Block = []
        growth_rate = 32
        self.Block.append(EncoderMeshBlock(self.EncodeChannels[0], self.EncodeChannels[1], growth_rate,
                                           self.num_clusters, max_iter=self.Niters[0],
                                           bn_momentum=self.bn_momentum))
        self.Block.append(EncoderMeshBlock(self.EncodeChannels[1], self.EncodeChannels[2], growth_rate,
                                           self.num_clusters, max_iter=self.Niters[1],
                                           bn_momentum=self.bn_momentum))
        self.Block.append(DualBlock(self.EncodeChannels[2], self.EncodeChannels[3], growth_rate,
                                    self.num_clusters, radius=self.radius[0], max_iter=self.Niters[2],
                                    bn_momentum=self.bn_momentum))
        self.Block.append(DualBlock(self.EncodeChannels[3], self.EncodeChannels[4], growth_rate,
                                    self.num_clusters, radius=self.radius[1], max_iter=self.Niters[3],
                                    bn_momentum=self.bn_momentum))
        self.Block.append(DualBlock(self.EncodeChannels[4], self.EncodeChannels[5], growth_rate,
                                    self.num_clusters, radius=self.radius[2], max_iter=self.Niters[4],
                                    bn_momentum=self.bn_momentum))

        # reducing channels of low-level features in the encoder
        self.Decode = []
        self.DecodeChannels = [self.EncodeChannels[-1], 128, 128, 96, 96, 96]
        for k in range(self.num_blocks):
            m = self.num_blocks - k - 1
            self.Decode.append(PerItemConv3d(self.EncodeChannels[m]+self.DecodeChannels[k],
                                             self.DecodeChannels[k+1], bn_momentum=self.bn_momentum))
            print(self.EncodeChannels[m]+self.DecodeChannels[k], self.DecodeChannels[k+1])

        # mesh pooling, unpooling configuration
        self.Mesh_Pool = mesh_pool(method='max')
        self.Mesh_Unpool = mesh_unpool()

        # classifier
        if self.predFacet:
            self.V2F0 = V2FConv3d(96, 96)
        self.Predict = PerItemConv3d(96, self.num_class, with_bn=False, activation_fn=None)

    def extract_facet_geometric_features(self, vertex, face, normals):
        face = face.to(torch.int)
        V1 = vertex[face[:,0],:]
        V2 = vertex[face[:,1],:]
        V3 = vertex[face[:,2],:]

        Height_features = torch.stack([V1[:,2], V2[:,2], V3[:,2]], dim=1)

        Delta12 = V2[:,:3] - V1[:,:3]
        Delta23 = V3[:,:3] - V2[:,:3]
        Delta31 = V1[:,:3] - V3[:,:3]

        # length of edges
        L12 = torch.norm(Delta12, dim=-1, keepdim=True)
        L23 = torch.norm(Delta23, dim=-1, keepdim=True)
        L31 = torch.norm(Delta31, dim=-1, keepdim=True)
        Length_features = torch.cat([L12, L23, L31], dim=-1)

        # angles between edges
        Theta1 = torch.sum(Delta12*(-Delta31), dim=-1, keepdim=True)/(L12*L31)
        Theta2 = torch.sum((-Delta12)*Delta23, dim=-1, keepdim=True)/(L12*L23)
        Theta3 = torch.sum((-Delta23)*Delta31, dim=-1, keepdim=True)/(L23*L31)
        Theta_features = torch.cat([Theta1, Theta2, Theta3], dim=-1)

        if self.useHeight:
            geometry_features = torch.cat([Length_features, Theta_features,
                                           normals, Height_features], dim=-1)
        else:
            geometry_features = torch.cat([Length_features, Theta_features,
                                           normals], dim=-1)
        return geometry_features

    @staticmethod
    def get_face_normals(geometry_in, shuffle_normals=None):
        if shuffle_normals:
            sign = torch.rand(geometry_in.shape[0], 1)
            sign = torch.where(sign<0.5, -1.0, 1.0)
            face_normals = sign*geometry_in[:,:3]
        else:
            face_normals = geometry_in[:,:3]
        return face_normals

    def __call__(self, vertex_in, face_in, nv_in, mf_in, shuffle_normals=True):
        vertex_input = vertex_in
        vertex_in = vertex_in[:,:3]

        print("build mesh hierarchy")
        mesh_hierarchy = self.build_mesh_hierarchy(vertex_in, face_in, nv_in, mf_in)
        print("done with mesh hierarchy")
        # import open3d as o3d
        # for i in range(6):
        #     vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[i][:4]
        #     mesh = o3d.geometry.TriangleMesh()
        #     mesh.vertices = o3d.utility.Vector3dVector(vertex_in[:,:3].numpy())
        #     mesh.triangles = o3d.utility.Vector3iVector(face_in.numpy())
        #     mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_in[:,3:].numpy())
        #     print(vertex_in.shape, face_in.shape)
        #     o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

        decoder_helper = []
        print("Initial Conv")
        # ============================================Initial Conv==============================================
        vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[0][:4]
        full_vt_map = torch.arange(vertex_in.shape[0]).to(self.device).to(torch.int)
        print("face_type",face_in.dtype)
        print("vt_map_type",full_vt_map.dtype)
        print("vertex_in_type",vertex_in.dtype)
        full_nf_count = meshUtil.count_vertex_adjface(face_in, full_vt_map, vertex_in)
        face_normals = self.get_face_normals(geometry_in, shuffle_normals=shuffle_normals)
        facet_geometrics = self.extract_facet_geometric_features(vertex_input, face_in, face_normals)
        filt_coeff = self.cluster[0](face_normals=face_normals)
        feats = self.Conv0(facet_geometrics, face_in, full_nf_count, full_vt_map, filt_coeff)
        # ===============================================The End================================================

        # ============================================Encoder Flow==============================================
        for k in range(self.num_blocks):
            # block computation:
            print("Encoder no.",k)
            vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[k][:4]
            full_vt_map = torch.arange(vertex_in.shape[0]).to(torch.int)
            full_nf_count = meshUtil.count_vertex_adjface(face_in, full_vt_map, vertex_in)
            face_normals = self.get_face_normals(geometry_in, shuffle_normals=shuffle_normals)
            filt_coeff = self.cluster[k](face_normals=face_normals)
            feats = self.Block[k](feats, vertex_in, face_in, full_nf_count, full_vt_map, filt_coeff, nv_in)

            if k<(self.num_blocks-1):
                decoder_helper.append(feats)
                vertex_out, _, _, _, _, vt_replace, vt_map = mesh_hierarchy[k+1]
                feats = self.Mesh_Pool(feats, vt_replace, vt_map, vertex_out)  # mesh pool
        # ===============================================The End================================================

        # ===============================combine with low-level encoder features================================
        for k in range(self.num_blocks-1):
            # upsampling and feature combination with Encoder features
            it = -k + (self.num_blocks-1)
            vt_replace, vt_map = mesh_hierarchy[it][-2:]
            low_level_feats = decoder_helper[it-1]
            upsampled_feats = self.Mesh_Unpool(feats, vt_replace, vt_map)
            feats = torch.concat([low_level_feats, upsampled_feats], dim=-1)
            feats = self.Decode[k](feats)
        # ===============================================The End================================================

        if self.predFacet:
            face_in = mesh_hierarchy[0][1]
            feats = self.V2F0(feats, face_in)
        pred_logits = self.Predict(feats)

        return pred_logits