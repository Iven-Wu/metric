import os
import pdb
import warnings
warnings.filterwarnings("ignore")

import sys
import copy
import random
import numpy as np
from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader

import trimesh
from PIL import Image
# import soft_renderer as sr
from ChamferDistance.chamfer3D.dist_chamfer_3D import chamfer_3DDist
# from chamferdist import ChamferDistance

from IOU3D import IIOU
from imageio import imread

from lietorch import SO3, SE3, LieGroupParameter

cham_loss = chamfer_3DDist()

def read_obj(obj_path, for_open_mesh=False):
    with open(obj_path) as file:
        flag = 0
        points = []
        normals = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == 'o' and flag == 0:
                flag = 1
                continue
            elif strs[0] == 'o' and flag == 1:
                break
            if strs[0] == 'v':
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))

            if strs[0] == 'vn':
                normals.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == 'f':
                single_line_face = strs[1:]

                f_co = []
                for sf in single_line_face:
                    face_tmp = sf.split('/')[0]
                    f_co.append(face_tmp)
                if for_open_mesh == False:
                    if len(f_co) == 3:
                        faces.append((int(f_co[0]), int(f_co[1]), int(f_co[2])))
                    elif len(f_co) == 4:
                        faces.append((int(f_co[0]), int(f_co[1]), int(f_co[2])))
                        faces.append((int(f_co[0]), int(f_co[1]), int(f_co[3])))
                        faces.append((int(f_co[1]), int(f_co[2]), int(f_co[3])))
                        faces.append((int(f_co[0]), int(f_co[3]), int(f_co[2])))
                else:
                    faces.append([int(ver) for ver in f_co])

    points = np.array(points)

    normals = np.array(normals)
    faces = np.array(faces) -1
    return points, normals, faces


def LBS(x, W1, T, R):
    # pdb.set_trace()
    bx = (T @ x.T).permute(0, 3, 1, 2)
    wbx = W1[None, :, :, None] * bx
    wbx = wbx.permute(0, 2, 1, 3)
    wbx = wbx.sum(1, keepdim=True)
    wbx = (R @ (wbx[:, 0].permute(0, 2, 1))).permute(0, 2, 1)

    return wbx

def load_data(animal_name='aardvark_female',frame=40):
    data_root = '/scratch/users/yuefanw/lasr/raw_log/{}-5/save'.format(animal_name)
    # gen_p,gen_n,gen_f = read_obj(os.path.join(data_root,'pred0.ply'))

    gt_root = '/scratch/users/yuefanw/version9/{}/'.format(animal_name)

    skinning = np.load(os.path.join(data_root, 'skin.npy'))
    source_cam_path = '{}/cam0.txt'.format(data_root)
    cam_mat = np.loadtxt(source_cam_path)
    source_focal = cam_mat[-1, 0]

    s_mesh = trimesh.load(os.path.join(data_root, 'pred0.ply'),process=False)
    t_mesh = trimesh.load(os.path.join(gt_root, 'frame_{:06d}.obj'.format(frame + 1)),process=False)
    t_info = np.load(os.path.join(gt_root, 'info', '{:04d}.npz'.format(frame + 1)))

    t_mesh.vertices = t_mesh.vertices[:, [0, 2, 1]]
    t_mesh.vertices[:, 1] *= -1

    t_cam_rot = t_info['cam_rot']
    t_cam_loc = t_info['cam_loc']
    t_mesh.vertices = t_mesh.vertices @ t_cam_rot
    t_mesh.vertices[:, -1] += np.linalg.norm(t_cam_loc)

    target_focal = t_info['intrinsic_mat'][0, 0]

    s_mesh.vertices[:, [1]] *= -1
    s_mean = np.mean(s_mesh.vertices, axis=0)
    s_mesh.vertices -= s_mean
    s_mesh.vertices *= source_focal / target_focal
    s_mesh.vertices += s_mean

    s_mesh.vertices[:, 1] -= s_mesh.vertices[:, 1].min()
    t_mesh.vertices[:, 1] -= t_mesh.vertices[:, 1].min()

    init_scale = t_mesh.vertices[:, -1].mean() / s_mesh.vertices[:, -1].mean()
    s_mesh.vertices *= (init_scale)

    s_mesh.vertices -= np.mean(t_mesh.vertices,axis=0)
    t_mesh.vertices -= np.mean(t_mesh.vertices,axis=0)
    #############################################

    ### root transformation
    # R = torch.eye(4).float().cuda()
    # ###
    # T = torch.eye(4).float().cuda().view(-1,4,4).repeat(skinning.shape[0],1,1)
    #
    # pdb.set_trace()
    return s_mesh,t_mesh,skinning

def minimize_chamfer(animal='aardvark_female',frame=40):
    s_mesh,t_mesh,skinning = load_data(animal_name=animal, frame=40)
    bone_number = skinning.shape[0]
    s_point_number = skinning.shape[1]
    t_point_number = t_mesh.vertices.shape[0]
    t_point = torch.tensor(t_mesh.vertices)[None,:].float().cuda()

    # pdb.set_trace()
    x = torch.tensor(np.array(s_mesh.vertices)).cuda()
    # pdb.set_trace()
    # pdb.set_trace()
    x = torch.cat([x,torch.ones((x.shape[0],1)).cuda()],dim=1).float()

    W = torch.tensor(skinning,device='cuda').T.float()

    Frames = 1
    p1 = torch.randn((Frames, bone_number + 1, 7), requires_grad=True, device="cuda")
    p1_init = np.array([0., 0., 0., 0., 0., 0., 1.])
    with torch.no_grad():
        for f in range(Frames):
            for b in range(bone_number):
                p1[f, b] = torch.tensor(p1_init)
        for f in range(Frames):
            p1[f, -1][:3] = torch.tensor([0, 0, 0. * f],
                                         dtype=torch.float64)  # Animal translation (0.01 y-axis for each frame)
            p1[f, -1][3:] = 0.

    # pdb.set_trace()

    new_R0 = SE3.InitFromVec(p1)
    R0 = new_R0.matrix().clone().detach()

    optimizer = optim.Adam([p1], lr=5e-3)


    for epoch in range(10):
        T = torch.randn(Frames, bone_number, 4, 4).cuda()  # T refers to bone transformations
        R = torch.randn(Frames, 4, 4).cuda()  # R refers to the root transformation
        # for idx, gt_idx in enumerate(index):
        for idx,gt_idx in enumerate(range(Frames)):
            R[idx] = SE3.InitFromVec(p1[gt_idx, -1]).matrix()
            for b in range(bone_number):
                R_temp = SE3.InitFromVec(p1[gt_idx, b])
                T[idx, b] = R_temp.matrix()

        # pdb.set_trace()
        wbx = LBS(x, W, T, R)

        # pdb.set_trace()

        optimizer.zero_grad()

        # pdb.set_trace()
        c_loss = cham_loss(wbx[:,:,:-1],t_point)[0].mean()
        print(c_loss.item())
        c_loss.backward()
        optimizer.step()

    s_mesh.vertices = wbx[0,:,:-1].detach().cpu().numpy()
    s_mesh.export('s_ac.obj')



if __name__ =='__main__':
    # load_data()
    minimize_chamfer()