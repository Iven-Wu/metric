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

from scipy.spatial import cKDTree as KDTree
import trimesh

from tqdm import tqdm

from queue import Queue
import copy
import json

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

def forward_kinematic(vert_pos, skin_weight, bones_len_scale, R, T, ske, json_data, mesh_scale):
    # all_bones = json_data['group_name']
    all_bones = list(ske.keys())

    all_bones.remove('def_c_root_joint')

    # all_bones = list(ske.keys())
    bones_dict = {}  # record index for each bone
    Rs_dict = {}  # record transformations of all parents for each bone
    for ind, bone in enumerate(all_bones):
        bones_dict[bone] = ind
        Rs_dict[bone] = []

    final_pos = torch.zeros_like(vert_pos[None].repeat(R.shape[0], 1, 1), requires_grad=True, device="cuda")  # final position
    root = ske['def_c_root_joint']
    q = Queue(maxsize=(len(bones_dict.keys()) + 1))
    q1 = Queue(maxsize=(len(bones_dict.keys()) + 1))
    root['new_head'] = root['head']  # new head/tail are the postion of head/tail after deformation
    root['new_tail'] = root['tail']
    root['tail_before_R'] = root['tail']
    for child in root['children']:
        if not child in all_bones:
            continue
        q.put(child)

    bone_sequence = []
    while not q.empty():  # record transformations of all parents for each bone in Rs_dict
        joint_name = q.get()
        bone_sequence.append(joint_name)
        bone_ind = bones_dict[joint_name]
        joint = ske[joint_name]
        for child in joint['children']:
            if not child in all_bones:
                continue
            q.put(child)
            Rs_dict[child].extend((Rs_dict[joint_name] + [bone_ind]))

    for child in root['children']:
        if not child in all_bones:
            continue
        q.put(child)

    while not q.empty():  # Apply transformation

        joint_name = q.get()
        joint = ske[joint_name]
        all_R_inds = Rs_dict[joint_name]
        bone_ind = bones_dict[joint_name]
        bone_scale = bones_len_scale[bone_ind]
        if "scale_head" not in joint.keys():
            joint['scale_head'] = joint['head'] * mesh_scale.clamp(0.15, 8).item()
            joint['scale_tail'] = joint['tail'] * mesh_scale.clamp(0.15, 8).item()
        head, tail = torch.from_numpy(joint['scale_head']).float().cuda(), torch.from_numpy(joint['scale_tail']).float().cuda()
        origin_local_pos = vert_pos - head
        bone_vec = tail - head
        endpoint_weight = (torch.mul(vert_pos - head, bone_vec).sum(1) / (bone_vec ** 2).sum().item()).clamp(0., 1.)

        if "new_head" not in joint.keys():
            joint['new_head'] = joint['scale_head'][:, None].repeat(R.shape[0], 1).transpose(1, 0)
            joint['new_tail'] = joint['scale_tail'][:, None].repeat(R.shape[0], 1).transpose(1, 0)
            joint['tail_before_R'] = joint['new_tail'] # 'tail_before_R' saves the position of bone tail before the bone itself is transformed

        new_head, new_tail = torch.from_numpy(joint['new_head']).float().cuda(), torch.from_numpy(
                joint['new_tail']).float().cuda()

        updated_vert_pos = vert_pos[None].repeat(R.shape[0], 1, 1).clone()
        if not len(all_R_inds) == 0:
            for R_ind in all_R_inds:  # apply transformations of parent skeletons
                R_bone_scale = bones_len_scale[R_ind]
                R_origin_head, R_origin_tail = torch.from_numpy(
                    ske[all_bones[R_ind]]['scale_head']).float().cuda(), torch.from_numpy(
                    ske[all_bones[R_ind]]['scale_tail']).float().cuda()
                R_head, R_tail = torch.from_numpy(ske[all_bones[R_ind]]['new_head']).float().cuda(), torch.from_numpy(
                    ske[all_bones[R_ind]]['tail_before_R']).float().cuda()  # take bone head as rotation center

                R_endpoint_weight = (torch.mul((vert_pos - R_origin_head), (R_origin_tail - R_origin_head)).sum(1) / (
                            (R_origin_tail - R_origin_head) ** 2).sum().item()).clamp(0., 1.)
                local_pos_for_update = updated_vert_pos - R_head[:, None]  # local position
                local_pos_for_update += (R_bone_scale - 1) * (R_endpoint_weight[:, None, None] * (R_tail - R_head)[None].repeat(vert_pos.shape[0], 1, 1)).permute(1, 0, 2)
                updated_local_vert_pos = R[:, [R_ind]].act(local_pos_for_update)  # transform based on lietorch
                updated_vert_pos = R_head[:, None] + updated_local_vert_pos  # global position

        local_pos = updated_vert_pos - new_head[:, None]
        local_pos += (bone_scale - 1) * (endpoint_weight[:, None, None] * (new_tail - new_head)[None].repeat(vert_pos.shape[0], 1, 1)).permute(1, 0, 2) # bone scale
        # local_pos = updated_vert_pos - new_head[None]
        # local_pos += endpoint_weight[:, None] * (bone_scale - 1) * (new_tail - new_head)
        updated_local_pos = R[:, [bone_ind]].act(local_pos)
        final_pos = final_pos + (new_head[:, None] + updated_local_pos) * skin_weight[:, [bone_ind]]  # LBS

        if not len(joint['children']) == 0:  # update positions of new_head/new_tail for all child bones
            for child in joint['children']:
                if not child in all_bones:
                    continue
                q.put(child)
                q1.put(child)
            while not q1.empty():
                child_joint_name = q1.get()
                child_joint = ske[child_joint_name]
                if "scale_head" not in child_joint.keys():
                    child_joint['scale_head'] = child_joint['head'] * mesh_scale.clamp(0.15, 8).item()
                    child_joint['scale_tail'] = child_joint['tail'] * mesh_scale.clamp(0.15, 8).item()
                if "new_head" not in child_joint.keys():
                    child_joint['new_head'] = child_joint['scale_head'][:, None].repeat(R.shape[0], 1).transpose(1, 0)
                    child_joint['new_tail'] = child_joint['scale_tail'][:, None].repeat(R.shape[0], 1).transpose(1, 0)
                    child_joint['tail_before_R'] = child_joint['new_tail']
                c_head_old, c_tail_old = torch.from_numpy(child_joint['scale_head']).float().cuda(), torch.from_numpy(
                    child_joint['scale_tail']).float().cuda()
                child_head, child_tail = torch.from_numpy(child_joint['new_head']).float().cuda(), torch.from_numpy(
                    child_joint['new_tail']).float().cuda()
                local_head, local_tail = child_head[None] - new_head[None], child_tail[None] - new_head[None]
                endweight_head = (torch.mul(c_head_old - head, bone_vec).sum() / (bone_vec ** 2).sum()).clamp(0.,
                                                                                                              1.).item()
                endweight_tail = (torch.mul(c_tail_old - head, bone_vec).sum() / (bone_vec ** 2).sum()).clamp(0.,
                                                                                                              1.).item()
                local_head += endweight_head * (bone_scale - 1) * (new_tail - new_head)[None]
                local_tail += endweight_tail * (bone_scale - 1) * (new_tail - new_head)[None]
                updated_local_head = R[:, [bone_ind]].act(local_head.permute(1, 0, 2))
                updated_local_tail = R[:, [bone_ind]].act(local_tail.permute(1, 0, 2))
                child_joint['new_head'] = (updated_local_head[:, 0] + new_head).detach().cpu().numpy()
                child_joint['new_tail'] = (updated_local_tail[:, 0] + new_head).detach().cpu().numpy()
                if not len(child_joint['children']) == 0:
                    for child in child_joint['children']:
                        if not child in all_bones:
                            continue
                        q1.put(child)
        joint['tail_before_R'] = joint['new_tail']
        new_tail = (new_head + R[:, [bone_ind]].act((bone_scale * (new_tail - new_head)[:, None]))[:, 0]).detach().cpu().numpy()
        joint['new_tail'] = new_tail  # update tail position for the currently selected bone

    final_pos = T.act(final_pos.clone())
    return final_pos

def LBS(x, W1, T, R):
    bx = (T @ x.T).permute(0, 3, 1, 2)
    wbx = W1[None, :, :, None] * bx
    wbx = wbx.permute(0, 2, 1, 3)
    wbx = wbx.sum(1, keepdim=True)
    wbx = (R @ (wbx[:, 0].permute(0, 2, 1))).permute(0, 2, 1)

    return wbx


def load_data(animal_name='aardvark_female',frame=40):
    # data_root = '/scratch/users/yuefanw/lasr/raw_log/{}-5/save'.format(animal_name)
    # gen_p,gen_n,gen_f = read_obj(os.path.join(data_root,'pred0.ply'))
    data_root = '/scratch/users/yuefanw/test_optim_ske11/{}/Epoch_201'.format(animal_name)

    gt_root = '/scratch/users/yuefanw/version9/{}/'.format(animal_name)

    skinning = np.load(os.path.join(data_root, 'temparray','W1.npy'))

    ske_ori = np.load(os.path.join(data_root,'temparray','ske.npy'),allow_pickle=True).item()

    ske = {}
    for key in ske_ori.keys():
        if 'joey' in key:
            continue
        ske[key] = {}
        if key=='def_c_root_joint':
            ske[key]['head'] = ske_ori[key]['new_head']
            ske[key]['tail'] = ske_ori[key]['new_tail']
        else:
            ske[key]['head'] = ske_ori[key]['new_head'][0]
            ske[key]['tail'] = ske_ori[key]['new_tail'][0]
        ske[key]['parent'] = ske_ori[key]['parent']
        ske[key]['children'] = ske_ori[key]['children']

    # source_cam_path = '{}/cam0.txt'.format(data_root)
    # source_cam_path = '{}/temparray'
    # cam_mat = np.loadtxt(source_cam_path)
    source_cam_path = '{}/info/0001.npz'.format(gt_root)
    s_info = np.load(source_cam_path)
    source_focal = s_info['intrinsic_mat'][0, 0]
    s_cam_rot = s_info['cam_rot']
    s_cam_loc = s_info['cam_loc']


    s_mesh = trimesh.load(os.path.join(data_root, 'Frame1.obj'),process=False)
    s_p,s_n,s_f = read_obj(os.path.join(data_root,'Frame1.obj'))
    s_mesh.vertices = s_p
    s_mesh.faces = s_f

    t_mesh = trimesh.load(os.path.join(gt_root, 'frame_{:06d}.obj'.format(frame + 1)),process=False)
    t_info = np.load(os.path.join(gt_root, 'info', '{:04d}.npz'.format(frame + 1)))

    t_mesh.vertices = t_mesh.vertices[:, [0, 2, 1]]
    t_mesh.vertices[:, 1] *= -1

    t_cam_rot = t_info['cam_rot']
    t_cam_loc = t_info['cam_loc']
    t_mesh.vertices = t_mesh.vertices @ t_cam_rot
    t_mesh.vertices[:, -1] += np.linalg.norm(t_cam_loc)

    # pdb.set_trace()

    target_focal = t_info['intrinsic_mat'][0, 0]

    # s_mesh.vertices[:, [1]] *= -1
    s_mesh.vertices = s_mesh.vertices[:,[0,2,1]]
    s_mesh.vertices[:,1] *= -1


    s_mesh.vertices = s_mesh.vertices @ s_cam_rot
    s_mesh.vertices[:,-1] += np.linalg.norm(s_cam_loc)


    s_mean = np.mean(s_mesh.vertices, axis=0)
    s_mesh.vertices -= s_mean
    s_mesh.vertices *= source_focal / target_focal
    s_mesh.vertices += s_mean

    s_mesh.vertices[:, 1] -= s_mesh.vertices[:, 1].min()
    t_mesh.vertices[:, 1] -= t_mesh.vertices[:, 1].min()
    s_mesh.vertices[:,0] -= s_mesh.vertices[:,0].min()
    t_mesh.vertices[:,0] -= t_mesh.vertices[:,0].min()

    # pdb.set_trace()

    init_scale = t_mesh.vertices[:, -1].mean() / s_mesh.vertices[:, -1].mean()
    s_mesh.vertices *= (init_scale)

    s_mesh.vertices -= np.mean(t_mesh.vertices,axis=0)
    t_mesh.vertices -= np.mean(t_mesh.vertices,axis=0)

    # pdb.set_trace()
    # pdb.set_trace()
    #############################################

    ### root transformation
    # R = torch.eye(4).float().cuda()
    # ###
    # T = torch.eye(4).float().cuda().view(-1,4,4).repeat(skinning.shape[0],1,1)
    #
    # pdb.set_trace()
    return s_mesh,t_mesh,skinning,ske

def minimize_chamfer(animal='aardvark_female',frame=40):
    s_mesh,t_mesh,skinning,ske = load_data(animal_name=animal, frame=frame)
    bone_number = skinning.shape[0]
    s_point_number = skinning.shape[1]
    t_point_number = t_mesh.vertices.shape[0]
    t_point = torch.tensor(t_mesh.vertices)[None,:].float().cuda()

    x = torch.tensor(np.array(s_mesh.vertices)).cuda()

    x = torch.cat([x,torch.ones((x.shape[0],1)).cuda()],dim=1).float()

    W = torch.tensor(skinning,device='cuda').float()

    Frames = 1
    p1 = torch.randn((Frames, bone_number, 4), requires_grad=True, device="cuda")
    p2 = torch.randn((Frames, 1, 7), requires_grad=True, device="cuda")
    p1_init = np.array([0., 0., 0., 1.])
    p2_init = np.array([0., 0., 0., 0., 0., 0., 1.])
    with torch.no_grad():
        for f in range(Frames):
            for b in range(bone_number):
                p1[f, b] = torch.tensor(p1_init)

        for f in range(Frames):
            p2[f, -1][:3] = torch.tensor([0, 0, 0.00 * f],
                                         dtype=torch.float64)  # Animal translation (0.01 y-axis for each frame)
            p2[f, -1][3:] = torch.tensor(p1_init)

    bones_len_scale = torch.ones((bone_number, 1),  device="cuda")

    mesh_scale = torch.ones((1),device="cuda")
    optimizer = optim.Adam([p1,p2], lr=5e-3)


    # pbar = tqdm(range(1000))

    # ske_ori =


    index = [0]

    pbar = tqdm(range(500))
    for epoch in pbar:

        SO3_R = SO3.InitFromVec(p1)
        SE3_T = SE3.InitFromVec(p2)

        # T = torch.randn(Frames, bone_number, 4, 4).cuda()  # T refers to bone transformations
        # R = torch.randn(Frames, 4, 4).cuda()  # R refers to the root transformation
        # # for idx, gt_idx in enumerate(index):
        # for idx,gt_idx in enumerate(range(Frames)):
        #     R[idx] = SE3.InitFromVec(p1[gt_idx, -1]).matrix()
        #     for b in range(bone_number):
        #         R_temp = SE3.InitFromVec(p1[gt_idx, b])
        #         T[idx, b] = R_temp.matrix()

        # wbx = LBS(x, W, T, R)

        json_data = None
        wbx = forward_kinematic(x[:, :-1].clone(), W.clone(), bones_len_scale, SO3_R[index[0]:index[-1] + 1],
                                SE3_T[index[0]:index[-1] + 1], ske, json_data, mesh_scale)

        # pdb.set_trace()
        optimizer.zero_grad()

        c_loss = cham_loss(wbx,t_point)[0].mean() + cham_loss(wbx,t_point)[1].mean()
        # pdb.set_trace()
        # print(c_loss.item())
        c_loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=c_loss.item())

    # print(c_loss.item())
    s_mesh.vertices = wbx[0].detach().cpu().numpy()
    os.makedirs('cus_reanim',exist_ok=True)
    s_mesh.export('cus_reanim/{}_{}.obj'.format(animal,frame))

    return c_loss.item()



if __name__ =='__main__':
    # load_data()

    val_split_list = [
                    'aardvark_female',
                      'aardvark_juvenile',
                      'aardvark_male',
                      'african_elephant_female',
                      'african_elephant_male',
                      'african_elephant_juvenile',
                      'binturong_female',
                      'binturong_juvenile',
                      'binturong_male',
                      'grey_seal_female',
                      'grey_seal_juvenile',
                      'grey_seal_male',
                      'bonobo_juvenile',
                      'bonobo_male',
                      'bonobo_female',
                      'polar_bear_female',
                      'polar_bear_juvenile',
                      'polar_bear_male',
                      'gray_wolf_female',
                      'gray_wolf_juvenile',
                      'gray_wolf_male',
                      'common_ostrich_female',
                      'common_ostrich_juvenile',
                      'common_ostrich_male'
                      ]
    frame_list = [40, 60, 80, 100, 120, 140]

    general_record = {}
    for frame in frame_list:

        cd_record = []
        for animal in val_split_list:

            print("Animal {}, frame {}".format(animal,frame))
            min_cd = minimize_chamfer(animal=animal,frame=frame)
            cd_record.append(min_cd)
        general_record[frame] = np.array(cd_record).mean()

    print(general_record)