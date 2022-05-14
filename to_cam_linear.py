import trimesh
import numpy as np
import os
import pdb
from tqdm import tqdm
from pdb import set_trace as bp
from IOU3D import IIOU

import argparse
iiou = IIOU()

def get_cam_IOU(animal_name):




    # for animal_name in val_animal_list:
    out_dir = 'viser_raw_IOU_camco/'
    os.makedirs(out_dir, exist_ok=True)

    print("Calculating for animal ",animal_name)
    iou_list = []
    iou_sum = 0
    sample_num = 0
    for frame in tqdm(range(30)):
        source_path = '/home/yuefanw/scratch/lasr/log/{}-5/save/pred{}.ply'.format(animal_name,frame)
        source_cam_path = '/home/yuefanw/scratch/lasr/log/{}-5/save/cam{}.txt'.format(animal_name, frame)

        # source_path = '/home/yuefanw/scratch/viser-release/log/{}-6/save/{}-vp1pred{}.obj'.format(animal_name,animal_name,frame)
        # source_cam_path = '/home/yuefanw/scratch/viser-release/log/{}-6/save/cam{}.txt'.format(animal_name,animal_name,frame)

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

        # scale_tmp = 4 / np.max(mesh.extents)
        matrix = np.eye(4)
        matrix[:3, :3] *= target_focal/source_focal
        s_mesh.apply_transform(matrix)


        s_mesh.vertices[:,1] -= s_mesh.vertices[:,1].mean()
        t_mesh.vertices[:,1] -= t_mesh.vertices[:,1].mean()

        s_mesh_tmp = s_mesh.copy()
        t_mesh_tmp = t_mesh.copy()


        init_scale =  t_mesh_tmp.vertices[:,-1].mean()/s_mesh_tmp.vertices[:,-1].mean()

        factor_list = [0.9,0.95,1,1.05,1.1]


        tmp_record = []
        for factor in factor_list:

            s_mesh_tmp = s_mesh.copy()
            t_mesh_tmp = t_mesh.copy()


            s_mesh_tmp.vertices *= (init_scale*factor)

            iou1 = iiou.compute_metric_obj(t_mesh_tmp, s_mesh_tmp,all_center=True)

            tmp_record.append(iou1)
            # print('factor ',iou1)


        iou = np.array(tmp_record).max()

        iou_list.append(iou)
        iou_sum += iou
        sample_num += 1



        # s_mesh.vertices[:,-1] -= s_mesh.vertices[:,-1].mean()
        # t_mesh.vertices[:,-1] -= t_mesh.vertices[:,-1].mean()


        # iou = iiou.compute_metric_obj(s_mesh, t_mesh)
        # iou_list.append(iou)
        # iou_sum += iou
        # sample_num += 1

    print('Average IOU', np.array(iou_list).mean())
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
    index = args.index
    animal_name = val_split_list[index]
    get_cam_IOU(animal_name)
    # for animal_name in val_split_list:
    #     icp_get_new(animal_name)



















