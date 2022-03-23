
import warnings
warnings.filterwarnings("ignore")
import argparse, time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from datetime import datetime
import picasso.mesh.utils as meshUtil
from picasso.augmentor import Augment
from picasso.models.shape_seg import PicassoNetII

from picasso.mesh.dataset import MeshDataset, default_collate_fn
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
import glob 
from writer import Writer
from sys import platform

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str, default="debug", help='path to tfrecord data')
parser.add_argument('--lm_ids',nargs='+', type=int, default= [0], help='number of cluster components')
parser.add_argument('--user',type=str, default="s183983", help='path to tfrecord data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../checkpoints', help='Log dir [default: ../log_shapenetcore]')
parser.add_argument('--ckpt_epoch', type=int, default=None, help='epoch model to load [default: None]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
parser.add_argument('--max_num_vertices', type=int, default=1500000, help='maximum vertices allowed in a batch')
parser.add_argument('--num_clusters', type=int, default=27, help='number of cluster components')
opt = parser.parse_args()

LOG_DIR = os.path.join(opt.log_dir,opt.name)
if not os.path.exists(opt.log_dir): os.mkdir(opt.log_dir)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# os.system('cp train_ShapeNetCore.py %s'%(LOG_DIR))   # bkp of train procedure
# os.system('cp picasso/models/shape_cls.py %s'%(LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')

NUM_CLASSES = 55


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


class TransformTrain(object):
    def __init__(self, prob=0.5, num_classes=None, drop_rate=0.1):
        self.prob = prob
        self.num_classes = num_classes
        self.drop_rate = drop_rate     # the rate of dropping vertices
        

    def augment_fn(self, vertex, face, texture=None, vertex_label=None, face_label=None):
        assert(vertex.shape[1]==3)
        vertex = Augment.flip_point_cloud(vertex, prob=self.prob)
        vertex = Augment.random_scale_point_cloud(vertex, scale_low=0.5, scale_high=1.5, prob=self.prob)
        vertex = Augment.rotate_point_cloud(vertex, upaxis=3, prob=self.prob)
        vertex = Augment.rotate_perturbation_point_cloud(vertex, prob=self.prob)
        vertex = Augment.jitter_point_cloud(vertex, sigma=0.01, prob=self.prob)
        if texture is not None:
            texture = Augment.random_drop_color(texture, prob=self.prob)
            texture = Augment.shift_color(texture, prob=self.prob)
            texture = Augment.jitter_color(texture, prob=self.prob)
            texture = Augment.auto_contrast_color(texture, prob=self.prob)
            vertex  = torch.cat([vertex, texture], dim=1)    # concat texture back
        if (self.drop_rate>.0) and (self.drop_rate<1.):
            vertex, face, label, \
            face_mask = Augment.random_drop_vertex(vertex, face, vertex_label, face_label,
                                                   drop_rate=self.drop_rate, prob=self.prob)
        else:
            label = vertex_label if face_label is None else face_label

        return vertex, face, label

    def __call__(self, args):
        vertex, face, label = args
        vertex, face, label = self.augment_fn(vertex[:,:3], face, vertex[:,3:], vertex_label=label)


        face_index = face.to(torch.long)
        face_texture = torch.cat([vertex[face_index[:,0],3:],
                                  vertex[face_index[:,1],3:],
                                  vertex[face_index[:,2],3:]], dim=1)
        bary_coeff = torch.eye(3).repeat([face.shape[0], 1])
        num_texture = 3*torch.ones(face.shape[0], dtype=torch.int)
        vertex = vertex[:,:3]

        num_labelled = torch.sum(((label>=0) & (label<self.num_classes)).to(torch.int))
        ratio = num_labelled/vertex.shape[0]
        if ratio<0.2:
            return None
        else:
            return [vertex, face, face_texture, bary_coeff, num_texture, label]
        # ===============================================================================================


class TransformTest(object):
    def __init__(self, prob=0.5, num_classes=None, drop_rate=0.1):
        self.prob = prob
        self.num_classes = num_classes
        self.drop_rate = drop_rate     # the rate of dropping vertices

    def __call__(self, args):
        vertex, face, label = args
    

        face_index = face.to(torch.long)
        face_texture = torch.cat([vertex[face_index[:,0],3:],
                                  vertex[face_index[:,1],3:],
                                  vertex[face_index[:,2],3:]], dim=1)
        bary_coeff = torch.eye(3).repeat([face.shape[0], 1])
        num_texture = 3*torch.ones(face.shape[0], dtype=torch.int)
        vertex = vertex[:,:3]

        num_labelled = torch.sum(((label>=0) & (label<self.num_classes)).to(torch.int))
        ratio = num_labelled/vertex.shape[0]
        if ratio<0.2:
            return None
        else:
            return [vertex, face, face_texture, bary_coeff, num_texture, label]
        # ===============================================================================================

class MyModel:
    def __init__(self, net, save_dir, 
                 start_epoch = 0, n_epochs = None):
        '''
            Setting all the variables for our model.
        '''
        super(MyModel, self).__init__()
        self.save_dir = save_dir
        self.device = "cuda:0"
        net.cuda(0)
        self.net = net.cuda()
        self.net.to(self.device)

        self.loss_metric = torch.nn.CrossEntropyLoss().to(self.device)
        self.start_epoch = start_epoch
        self.n_epochs = n_epochs

        
        #self.save_dir = join(opt.checkpoints_dir, opt.name)
        starter_learning_rate = 0.001
        decay_steps = 20000
        decay_rate = 0.5
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr = starter_learning_rate,
                                     eps = 1e-08,
                                     weight_decay= 0)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                                  gamma=decay_rate)
        self.loss = None
        if start_epoch > 0:
            self.load_network(start_epoch)
        self.writer = Writer(save_dir)
        

    # @tf.function(experimental_relax_shapes=True)
    def get_accuracy(self, pred, label):
        return (pred.cpu().argmax(1) == label).float().mean()
    
    def train_step(self, meshes, labels):
        """
        Update parameters in network for a single epoch

        Parameters
        ----------
        meshes : TYPE
            DESCRIPTION.
        labels : TYPE
            DESCRIPTION.
            
        Input from batch as
            meshes, labels = data[:4], data[-1]
            
        Returns
        -------
        None.

        """
        vertex_in, face_in, nv_in, mf_in = meshes
        #vertex_in = vertex_in[:tf.reduce_sum(nv_in),:] # change input axis-0 size to None
        #face_in = face_in[:tf.reduce_sum(mf_in),:]     # change input axis-0 size to None
        self.optimizer.zero_grad()        
        out = self.net(vertex_in, face_in, nv_in, mf_in)
        self.loss = self.train_loss(out, labels)
        self.loss.backward()
        self.optimizer.step()
        train_acc = self.get_accuracy(out,labels)
        
        # with tf.GradientTape() as tape:
        #     pred_logits = self.model(vertex_in, face_in, nv_in, mf_in, tf.constant(True))
        #     onehot_labels = tf.one_hot(labels, depth=NUM_CLASSES)
        #     loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_logits, labels=onehot_labels)

        # gradients = tape.gradient(loss, self.trainable_variables)
        # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # self.train_loss(loss)
        # self.train_metric(y_true=onehot_labels, y_pred=pred_logits)

    # @tf.function(experimental_relax_shapes=True)
    def test_step(self, meshes, labels):
        vertex_in, face_in, nv_in, mf_in = meshes
        #vertex_in = vertex_in[:tf.reduce_sum(nv_in), :]  # change input axis-0 size to None
        #face_in = face_in[:tf.reduce_sum(mf_in), :]      # change input axis-0 size to None
        with torch.no_grad():
                out = self.net(vertex_in, face_in, nv_in, mf_in)
        # compute number of correct
                pred_class = out.data.max(1)[1]
                label_class = self.labels
                # self.export_segmentation(pred_class.cpu())
                acc = self.get_accuracy(pred_class, label_class)
        return acc
        # 
        # onehot_labels = tf.one_hot(labels, depth=NUM_CLASSES)
        # t_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_logits, labels=onehot_labels)

        # self.test_loss(t_loss)
        # self.test_metric(y_true=onehot_labels, y_pred=pred_logits)
        # return labels, pred_logits
    
    def fit(self, train_set, vali_set, batch_size, start_epoch = 0, n_epochs = 200):
        total_steps = 0
        print_freq = 20
        self.net.train()
        for epoch in range(start_epoch, n_epochs + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(train_set):
                iter_start_time = time.time()
                if total_steps % print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += batch_size
                epoch_iter += batch_size
                # meshes, labels = data
                meshes, labels = data[:4], data[-1] #change if using texture
                self.train_step(meshes, labels)


            if total_steps % opt.print_freq == 0:
                loss = self.net.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                self.writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                self.save_network('latest')
                self.save_network(epoch)
    
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

            
            self.lr_scheduler.step()
            acc = 0
            if epoch % 1 == 0:
                self.net.eval()
                for i, data in enumerate(vali_set):
                    meshes, labels = data
                    acc += self.test_step(meshes, labels).mean()
                self.writer.print_acc(acc, epoch)
                self.net.train()
            iter_data_time = time.time()
        
            
    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = os.path.join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

if __name__=='__main__':
    if platform == "win32":
        root = 'C:/Users/lowes/OneDrive/Skrivebord/DTU/8_Semester/Advaced_Geometric_DL/BU_3DFE_3DHeatmaps_crop_2/'
    else:
        root = "/scratch/s183983/data_cropped/" if opt.user=="s183983" \
            else "/scratch/s183986//BU_3DFE_3DHeatmaps_crop/"
    # json file_lists
    files_train = glob.glob(os.path.join(root,"train","*.vtk"))
    files_val = glob.glob(os.path.join(root,"val","*.vtk"))
    files_test = glob.glob(os.path.join(root,"test","*.vtk"))
    

    # build training set dataloader
    trainSet = MeshDataset(files_train, root,opt.lm_ids, rendered_data=True,
                           transform=TransformTrain(num_classes=len(opt.lm_ids)+1))
    trainLoader = DataLoader(trainSet, batch_size=opt.batch_size, shuffle=True,
                             collate_fn=default_collate_fn(opt.max_num_vertices))
    # build validation set dataloader
    valSet = MeshDataset(files_val, root,opt.lm_ids, rendered_data=True,
                         transform=TransformTest(num_classes=len(opt.lm_ids)+1))
    valLoader = DataLoader(valSet, batch_size=opt.batch_size, shuffle=True,
                           collate_fn=default_collate_fn(opt.max_num_vertices))

    # create model & Make a loss object
    # for i, data in enumerate(valLoader):
    #     print(data.shape)
    net = PicassoNetII(num_class=NUM_CLASSES, mix_components=opt.num_clusters, use_height=True)
    model = MyModel(net, LOG_DIR)
    
    model.fit(trainLoader,valLoader,opt.batch_size)
    
    
    
