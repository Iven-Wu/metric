import trimesh
import numpy as np
import os
import pdb
from tqdm import tqdm
from pdb import set_trace as bp
from IOU3D import IIOU

from ChamferDistance.chamfer3D.dist_chamfer_3D import chamfer_3DDist
import torch
import argparse
iiou = IIOU()

cham_loss = chamfer_3DDist()

def get_cam_IOU(animal_name):


    out_dir = 'viser_raw_IOU_camco/'
    os.makedirs(out_dir, exist_ok=True)

    print("Calculating for animal ",animal_name)
    iou_list = []
    chamfer_list = []
    iou_sum = 0
    sample_num = 0
    for frame in tqdm(range(30)):
    # for frame in tqdm(range(31,35)):
    #     source_path = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/pred{}.ply'.format(animal_name,frame)
    #     source_cam_path = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/cam{}.txt'.format(animal_name, frame)

        # source_path = '/home/yuefanw/scratch/lasr_re/raw_log/{}-re/save_new/pred{}.ply'.format(animal_name,frame)
        # source_cam_path = '/home/yuefanw/scratch/lasr_re/raw_log/{}-re/save_new/cam{}.txt'.format(animal_name, frame)
        source_path = '/home/yuefanw/scratch/viser-release/log/{}-6/save/{}-vp1pred{}.obj'.format(animal_name,animal_name,frame)
        source_cam_path = '/home/yuefanw/scratch/viser-release/log/{}-6/save/{}-cam{}.txt'.format(animal_name,animal_name,frame)

        cam_mat = np.loadtxt(source_cam_path)
        source_focal = cam_mat[-1,0]

        target_path = '/home/yuefanw/scratch/version9/{}/frame_{:06d}.obj'.format(animal_name,frame+1)

        # source_cam_path = '/home/yuefanw/scratch/lasr/log/{}-5/save/{}.txt'.format(animal_name,frame)
        target_info_path = '/home/yuefanw/scratch/version9/{}/info/{:04d}.npz'.format(animal_name,frame+1)
        t_info = np.load(target_info_path)

        s_mesh = trimesh.load(source_path)
        t_mesh = trimesh.load(target_path)


        # pdb.set_trace()

        t_mesh.vertices = t_mesh.vertices[:,[0,2,1]]
        t_mesh.vertices[:,1] *= -1

        #### gt_mesh_coord
        # t_info = np.load(target_info_path)
        t_cam_rot = t_info['cam_rot']
        t_cam_loc = t_info['cam_loc']
        t_mesh.vertices = t_mesh.vertices @ t_cam_rot
        t_mesh.vertices[:,-1] += np.linalg.norm(t_cam_loc)

        target_focal = t_info['intrinsic_mat'][0,0]

        #### pred_mesh_coord
        s_mesh.vertices[:,[1]] *= -1

        # s_mesh.scale(target_focal/source_focal,center=s_mesh.get_center())

        # pdb.set_trace()

        # scale_tmp = 4 / np.max(mesh.extents)
        # matrix = np.eye(4)
        # matrix[:3, :3] *= source_focal/target_focal
        #
        # s_mesh.apply_transform(matrix)

        s_mean = np.mean(s_mesh.vertices,axis=0)
        s_mesh.vertices -= s_mean
        s_mesh.vertices *= source_focal/target_focal
        s_mesh.vertices += s_mean



        s_mesh.vertices[:,1] -= s_mesh.vertices[:,1].min()
        t_mesh.vertices[:,1] -= t_mesh.vertices[:,1].min()

        s_mesh_tmp = s_mesh.copy()
        t_mesh_tmp = t_mesh.copy()


        init_scale =  t_mesh_tmp.vertices[:,-1].mean()/s_mesh_tmp.vertices[:,-1].mean()

        s_mesh.vertices *= (init_scale)

        # pdb.set_trace()
        ### icp, trans is result pcd, mat is rotation matrix
        mat, trans, cost = trimesh.registration.icp(s_mesh.vertices, t_mesh.vertices, max_iterations=1000)

        s_mesh.vertices = trans

        s_mesh_tmp = np.concatenate((np.array(s_mesh.vertices), np.ones((s_mesh.vertices.shape[0],1)) ),axis=1)
        rot_result = (mat@s_mesh_tmp.T).T

        # factor_list = [0.9,0.95,1,1.05,1.1]


        # tmp_record = []
        # for factor in factor_list:
        #
        #     s_mesh_tmp = s_mesh.copy()
        #     t_mesh_tmp = t_mesh.copy()
        #
        #
        #     s_mesh_tmp.vertices *= (init_scale*factor)
        #
        #     iou1 = iiou.compute_metric_obj(t_mesh_tmp, s_mesh_tmp,all_center=True)
        #
        #     tmp_record.append(iou1)

        # pdb.set_trace()

        chamfer_loss = cham_loss(torch.tensor(np.array(s_mesh.vertices))[None, :].float().cuda(),
                  torch.tensor(np.array(t_mesh.vertices))[None, :].float().cuda())[0].mean()

        iou = iiou.compute_metric_obj(t_mesh, s_mesh,all_center=True)
        iou_list.append(iou)

        chamfer_list.append(chamfer_loss.detach().cpu().numpy())

        sample_num += 1


        # iou = iiou.compute_metric_obj(s_mesh, t_mesh)
        # iou_list.append(iou)
        # sample_num += 1

    print('Average IOU', np.array(iou_list).mean())
    print('Average Chamfer', np.array(chamfer_list).mean())
    np.save(os.path.join(out_dir, '{}.npy'.format(animal_name)), np.array(iou_list))
            # pdb.set_trace()


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
    for animal_name in val_split_list:
        # index = args.index
        # animal_name = val_split_list[index]
        get_cam_IOU(animal_name)
    # for animal_name in val_split_list:
    #     icp_get_new(animal_name)



















