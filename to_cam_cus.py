import trimesh
import numpy as np
import os
import pdb
from tqdm import tqdm
from pdb import set_trace as bp
from IOU3D import IIOU
import torch
from ChamferDistance.chamfer3D.dist_chamfer_3D import chamfer_3DDist

import argparse
iiou = IIOU()


cham_loss = chamfer_3DDist()

def get_cam_IOU(animal_name):

    # for animal_name in val_animal_list:
    out_dir = 'cus_IOU_camco/'
    os.makedirs(out_dir, exist_ok=True)

    print("Calculating for animal ",animal_name)
    iou_list = []
    chamfer_list = []
    iou_sum = 0
    sample_num = 0
    for frame in tqdm(range(30)):
        # source_path = '/home/yuefanw/scratch/lasr/log/{}-5/save/pred{}.ply'.format(animal_name,frame)

        # source_path = '/scratch/users/yuefanw/test_optim_ske8/{}/Epoch121/Frame{}.obj'.format(animal_name,frame+1)
        source_path = '/scratch/users/yuefanw/test_optim_single/{}222/Epoch_120/Frame{}.obj'.format(animal_name,frame+1)

        target_path = '/home/yuefanw/scratch/version9/{}/frame_{:06d}.obj'.format(animal_name,frame+1)

        # source_cam_path = '/home/yuefanw/scratch/lasr/log/{}-5/save/{}.txt'.format(animal_name,frame)
        target_info_path = '/home/yuefanw/scratch/version9/{}/info/{:04d}.npz'.format(animal_name,frame+1)


        s_mesh = trimesh.load(source_path)
        t_mesh = trimesh.load(target_path)




        t_info = np.load(target_info_path)
        t_cam_rot = t_info['cam_rot']
        t_cam_loc = t_info['cam_loc']

        #### gt_mesh_coord
        t_mesh.vertices = t_mesh.vertices[:,[0,2,1]]
        t_mesh.vertices[:,1] *= -1
        t_mesh.vertices = t_mesh.vertices @ t_cam_rot
        t_mesh.vertices[:,-1] -= np.linalg.norm(t_cam_loc)



        #### to gt mesh coord
        # pdb.set_trace()
        s_mesh.vertices = s_mesh.vertices[:,[0,2,1]]
        s_mesh.vertices[:,1] *= -1
        s_mesh.vertices = s_mesh.vertices @ t_cam_rot
        s_mesh.vertices[:,-1] -= np.linalg.norm(t_cam_loc)


        # pdb.set_trace()

        #### pred_mesh_coord

        mat, trans, cost = trimesh.registration.icp(s_mesh.vertices, t_mesh.vertices, max_iterations=1000)

        s_mesh.vertices = trans

        # s_mesh.vertices[:,-1] -= s_mesh.vertices[:,-1].mean()
        # t_mesh.vertices[:,-1] -= t_mesh.vertices[:,-1].mean()

        # pdb.set_trace()

        chamfer_loss = cham_loss(torch.tensor(np.array(s_mesh.vertices))[None, :].float().cuda(),
                  torch.tensor(np.array(t_mesh.vertices))[None, :].float().cuda())[0].mean()

        iou = iiou.compute_metric_obj(t_mesh, s_mesh,all_center=True)
        iou_list.append(iou)
        iou_sum += iou

        chamfer_list.append(chamfer_loss.detach().cpu().numpy())
        sample_num += 1

    print('Average IOU', np.array(iou_list).mean())
    print('Average Chamfer', np.array(chamfer_list).mean())
    np.save(os.path.join(out_dir, '{}.npy'.format(animal_name)), np.array(iou_list))
            # pdb.set_trace()


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='instruct')
    parser.add_argument('--index',default=0,type=int)
    args = parser.parse_args()
    # animal_name = 'arctic_wolf_male'

    val_split_list = [
                    # 'aardvark_female',
                    #   'aardvark_juvenile',
                      'aardvark_male',
                      'african_elephant_female',
                    #   'african_elephant_male',
                    #   'african_elephant_juvenile',
                    #   'binturong_female',
                    #   'binturong_juvenile',
                    #   'binturong_male',
                      'grey_seal_female',
                      'grey_seal_juvenile',
                      'grey_seal_male',
                      # 'bonobo_juvenile',
                      # 'bonobo_male',
                      # 'bonobo_female',
                      # 'polar_bear_female',
                      # 'polar_bear_juvenile',
                      # 'polar_bear_male',
                      # 'gray_wolf_female',
                      # 'gray_wolf_juvenile',
                      # 'gray_wolf_male',
                      # 'common_ostrich_female',
                      # 'common_ostrich_juvenile',
                      # 'common_ostrich_male'
                      ]
    # index = args.index
    for index in range(len(val_split_list)):
        animal_name = val_split_list[index]
        get_cam_IOU(animal_name)

    # for animal_name in val_split_list:
    #     icp_get_new(animal_name)
