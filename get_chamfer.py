import trimesh
import numpy as np
import os
import pdb
from tqdm import tqdm
from pdb import set_trace as bp
import torch
import argparse

from ChamferDistance.chamfer3D.dist_chamfer_3D import chamfer_3DDist

cham_loss = chamfer_3DDist()

# cham_loss(x[None, :, :-1], x_reverse[None, :, :-1])[0].mean()

def center( mesh):
    bb = mesh.bounds
    b_center = np.mean(bb, axis=0)
    mesh.vertices -= b_center
    return mesh

def get_chamfer(animal_name):

    # for animal_name in val_animal_list:
    out_dir = 'viser_raw_chamfer_camco/'
    os.makedirs(out_dir, exist_ok=True)

    print("Calculating for animal ",animal_name)
    chamfer_list = []
    chamfer_sum = 0
    sample_num = 0
    for frame in tqdm(range(30)):
        # source_path = '/home/yuefanw/scratch/lasr/log/{}-5/save/pred{}.ply'.format(animal_name,frame)
        source_path = '/home/yuefanw/scratch/viser-release/log/{}-6/save/{}-vp1pred{}.obj'.format(animal_name,
                                                                                                  animal_name, frame)
        target_path = '/home/yuefanw/scratch/version9/{}/frame_{:06d}.obj'.format(animal_name,frame+1)

        # source_cam_path = '/home/yuefanw/scratch/lasr/log/{}-5/save/{}.txt'.format(animal_name,frame)
        target_info_path = '/home/yuefanw/scratch/version9/{}/info/{:04d}.npz'.format(animal_name,frame+1)


        s_mesh = trimesh.load(source_path)
        t_mesh = trimesh.load(target_path)

        t_mesh.vertices = t_mesh.vertices[:,[0,2,1]]
        t_mesh.vertices[:,1] *= -1

        #### gt_mesh_coord
        t_info = np.load(target_info_path)
        t_cam_rot = t_info['cam_rot']
        t_cam_loc = t_info['cam_loc']
        t_mesh.vertices = t_mesh.vertices @ t_cam_rot
        t_mesh.vertices[:,-1] -= np.linalg.norm(t_cam_loc)

        #### pred_mesh_coord
        s_mesh.vertices[:,[1,2]] *= -1

        s_mesh.vertices[:,-1] -= s_mesh.vertices[:,-1].mean()
        t_mesh.vertices[:,-1] -= t_mesh.vertices[:,-1].mean()

        ####### center
        s_mesh = center(s_mesh)
        t_mesh = center(t_mesh)
        ########


        gt_extent = t_mesh.extents
        gt_x_extent = gt_extent[0]

        pred_extent = s_mesh.extents
        pred_x_extent = pred_extent[0]

        scale = gt_x_extent/pred_x_extent

        matrix = np.eye(4)
        matrix[:3,:3] *= scale
        s_mesh.apply_transform(matrix)

        # pdb.set_trace()

        # pdb.set_trace()
        chamfer_distance = cham_loss(torch.tensor(np.array(s_mesh.vertices))[None, :].float().cuda(), torch.tensor(np.array(t_mesh.vertices))[None, :].float().cuda())[0].mean()

        # pdb.set_trace()
        # pdb.set_trace()
        chamfer_list.append(chamfer_distance.item())
        chamfer_sum += chamfer_distance.item()
        sample_num += 1

    avg_chamfer = np.array(chamfer_list).mean()
    print('Average Chamfer {:.3f}'.format(avg_chamfer))

    return avg_chamfer


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='instruct')
    parser.add_argument('--index',default=0,type=int)
    args = parser.parse_args()
    # animal_name = 'arctic_wolf_male'

    val_split_list = ['aardvark_female',
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
    index = args.index
    animal_name = val_split_list[index]

    avg_record = []
    for animal_name in val_split_list:
        avg_chamfer = get_chamfer(animal_name)
        avg_record.append(avg_chamfer)

    total_avg = np.array(avg_record).mean()
    print('Total Average Chamfer', total_avg)