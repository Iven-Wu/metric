import time

import numpy as np
# import open3d as o3d
import trimesh
from pdb import set_trace as bp

from tqdm import tqdm
# import pandas as pd
import os
import argparse
from torch.utils.data import DataLoader
import torch

class IIOU():

    def __init__(self,):

        self.resolution = 128
        self.b_min = np.array([-2, -2, -2])
        self.b_max = np.array([5, 5, 5])
        self.transform = None
        self.num_samples = 2097152
        self.max_score = -100

        self.time_count = 0

    def create_grid(self, resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):

        coords = np.mgrid[:resX, :resY, :resZ]
        coords = coords.reshape(3, -1)
        coords_matrix = np.eye(4)
        length = b_max - b_min
        coords_matrix[0, 0] = length[0] / resX
        coords_matrix[1, 1] = length[1] / resY
        coords_matrix[2, 2] = length[2] / resZ
        coords_matrix[0:3, 3] = b_min
        coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
        if transform is not None:
            coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
            coords_matrix = np.matmul(transform, coords_matrix)
        coords = coords.reshape(3, resX, resY, resZ)
        return coords, coords_matrix

    def normalize(self, mesh):
        scale = 4 / np.max(mesh.extents)
        matrix = np.eye(4)
        matrix[:3, :3] *= scale
        mesh.apply_transform(matrix)

        return mesh

    def normalize_new(self,mesh_gt,mesh_pred):
        gt_extent = mesh_gt.extents
        gt_x_extent = gt_extent[0]

        pred_extent = mesh_pred.extents
        pred_x_extent = pred_extent[0]

        scale = gt_x_extent/pred_x_extent

        matrix = np.eye(4)
        matrix[:3,:3] *= scale
        mesh_pred.apply_transform(matrix)

        return mesh_pred

    def center(self,mesh):
        bb = mesh.bounds
        b_center = np.mean(bb,axis=0)
        mesh.vertices -= b_center
        return mesh

    def center_all(self,mesh_gt,mesh_pred):
        bb = mesh_gt.bounds
        b_center = np.mean(bb,axis=0)

        mesh_gt.vertices -= b_center
        mesh_pred.vertices -= b_center

        return mesh_gt,mesh_pred

    def compute_metric_obj(self,mesh_gt,mesh_pred,all_center=False):
        
        # if type(mesh_gt)==str:
        #     mesh_gt = trimesh.load(mesh_gt)
        # if type(mesh_tar)==str:
        #     mesh_tar = trimesh.load(mesh_tar)

        if not all_center:
            mesh_gt = self.center(mesh_gt)
            mesh_pred = self.center(mesh_pred)
        else:
            mesh_gt,mesh_pred = self.center_all(mesh_gt,mesh_pred)

        # mesh_gt = self.normalize(mesh_gt)
        # mesh_pred = self.normalize(mesh_pred)
        mesh_pred = self.normalize_new(mesh_gt,mesh_pred)

        coords, mat = self.create_grid(self.resolution, self.resolution, self.resolution,
                            self.b_min, self.b_max)
        points = coords.reshape([3, -1]).T

        pred_gt = mesh_gt.contains(points)
        pred_tar = mesh_pred.contains(points)
        intersection = np.logical_and(pred_gt,pred_tar).astype(np.int32).sum()
        union = np.logical_or(pred_gt,pred_tar).astype(np.int32).sum()

        iou = intersection/union

        del pred_tar, pred_gt, mesh_pred, mesh_gt, points

        return iou

if __name__ =='__main__':
    oo = IIOU()
    gt_mesh = '/scratch/users/yuefanw/version8/aardvark_male/frame_000001.obj'
    tar_mesh = '/scratch/users/yuefanw/ppp/results_recon_cpu123/aardvark_male/0001.obj'
    gt_mesh = tar_mesh
    iou = oo.compute_metric_obj(gt_mesh,tar_mesh)
    print(iou)




