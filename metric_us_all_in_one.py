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
import scipy
iiou = IIOU()

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
    faces = np.array(faces) - 1
    return points, normals, faces

def chamfer_dist(pt1, pt2):
    pt1 = pt1[np.newaxis, :, :]
    pt2 = pt2[:, np.newaxis, :]
    dist = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=2))
    min_left = np.mean(np.min(dist, axis=0))
    min_right = np.mean(np.min(dist, axis=1))
    #print(min_left, min_right)
    return (min_left + min_right) / 2

def read_gt_joints(animal_name,frame):

    # gt_mesh = trimesh.load()
    fpath3 = '/projects/perception/datasets/animal_videos/version9'
    ske_all = \
        np.load(os.path.join(fpath3, animal_name, 'skeleton', 'skeleton_all_frames.npy'), allow_pickle=True).item()

    gt_mesh = trimesh.load(os.path.join(fpath3,animal_name,'frame_{:06d}.obj'.format(frame+1)))

    ske = ske_all['frame_{:06d}'.format(frame + 1)]
    info_path = '/projects/perception/datasets/animal_videos/version9/{}/info/{:04d}.npz'.format(animal_name,
                                                                                                 frame + 1)
    gt_info = np.load(info_path)
    gt_cam_rot = gt_info['cam_rot']
    gt_cam_loc = gt_info['cam_loc']

    joint_list = []
    for key in ske.keys():
        if key == 'def_c_root_joint':
            continue

        joint_list.append(((ske[key]['head'] + ske[key]['tail']) / 2)[np.newaxis, :])

    # pdb.set_trace()
    joint_array = np.concatenate(joint_list, axis=0)

    gt_mesh.vertices = np.concatenate((gt_mesh.vertices))

    joint_array_new = joint_array @ gt_cam_rot
    joint_array_new[:, -1] += np.linalg.norm(gt_cam_loc)

    joint_array_new[:, 1] -= joint_array_new[:, 1].min()

    return joint_array_new


def get_skin_cost(s_mesh,animal_name):

    s_mesh_p = s_mesh.vertices

    # source_root = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/'.format(animal_name)
    # source_root = '/home/yuefanw/scratch/viser-release/log/{}-6/save/'.format(animal_name)
    source_root = '/home/yuefanw/scratch/test_optim_ske11/{}/Epoch_201/temparray'.format(animal_name)
    target_root = '/home/yuefanw/scratch/simplified_meshes/simplified_meshes/{}'.format(animal_name)

    target_info_path = '/projects/perception/datasets/animal_videos/version9/{}/info/0001.npz'.format(animal_name)
    t_info = np.load(target_info_path)

    t_cam_rot = t_info['cam_rot']
    t_cam_loc = t_info['cam_loc']
    # t_mesh.vertices = t_mesh.vertices @ t_cam_rot
    # t_mesh.vertices[:, -1] += np.linalg.norm(t_cam_loc)



    s_weight = np.load(os.path.join(source_root,'W1.npy')).T
    t_weight = np.load(os.path.join(target_root,'W1.npy')).T


    t_p,t_n,t_f = read_obj(os.path.join(target_root,'remesh.obj'))
    t_mesh_p = t_p

    t_mesh_p = t_mesh_p @ t_cam_rot
    t_mesh_p[:,-1] += np.linalg.norm(t_cam_loc)

    t_mesh_p[:,0] -= t_mesh_p[:,0].min()
    t_mesh_p[:,1] -= t_mesh_p[:,1].min()

    # pdb.set_trace()

    # t_mesh_p[:,0] -= t_mesh_p[:,0].min()
    # t_mesh_p[:,1] -= t_mesh_p[:,1].min()
    # s_mesh.vertices[:, 0] -= s_mesh.vertices[:, 0].min()
    # t_mesh.vertices[:, 0] -= t_mesh.vertices[:, 0].min()
    # s_mesh.vertices[:, 1] -= s_mesh.vertices[:, 1].min()
    # t_mesh.vertices[:, 1] -= t_mesh.vertices[:, 1].min()

    # if t_weight[-1].sum() < 0.1:
    #     t_weight
    while t_weight[-1].sum() < 0.1:
        t_weight = t_weight[:-1]

    # pdb.set_trace()

    cost_matrix = np.zeros((s_weight.shape[0],t_weight.shape[0]))

    # pdb.set_trace()

    for s_bone in range(s_weight.shape[0]):
        # s_weight[]
        # pdb.set_trace()
        s_index = np.where(s_weight[s_bone]>1e-4)[0]
        s_points_loc = s_mesh_p[s_index]

        for t_bone in range(t_weight.shape[0]):
            t_index = np.where(t_weight[t_bone]>1e-4)[0]
            t_points_loc = t_mesh_p[t_index]


            if t_points_loc.shape[0]==0:
                cost_matrix[s_bone,t_bone] = 100
            else:
                cost_matrix[s_bone,t_bone] = chamfer_dist(s_points_loc,t_points_loc)


    row_ind,col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

    # min_cost = cost_matrix[row_ind,col_ind].sum()
    min_cost = cost_matrix[row_ind, col_ind].mean()


    # pdb.set_trace()
    return min_cost

def get_cam_IOU(animal_name):


    out_dir = 'viser_raw_IOU_camco/'
    os.makedirs(out_dir, exist_ok=True)

    print("Calculating for animal ",animal_name)
    iou_list = []
    chamfer_list = []
    bone_cd_list =[]
    skin_list = []
    sample_num = 0
    for frame in tqdm(range(30)):
    # for frame in tqdm(range(31,35)):
    #     source_path = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/pred{}.ply'.format(animal_name,frame)
    #     source_cam_path = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/cam{}.txt'.format(animal_name, frame)
    #     s_joint_dir = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/cam_bone{}.ply'.format(animal_name, frame)

        # source_path = '/home/yuefanw/scratch/viser-release/log/{}-6/save/{}-vp1pred{}.obj'.format(animal_name,animal_name, frame)
        # source_cam_path = '/home/yuefanw/scratch/viser-release/log/{}-6/save/{}-cam{}.txt'.format(animal_name,animal_name, frame)
        # s_joint_dir = '/home/yuefanw/scratch/viser-release/log/{}-6/save/{}-bones{}.npy'.format(animal_name, animal_name,frame)

        source_path = '/home/yuefanw/scratch/test_optim_ske11/{}/Epoch_201'.format(animal_name)

        source_mesh_path = '{}/Frame{}.obj'.format(source_path,frame+1)
        source_cam_path = '/projects/perception/datasets/animal_videos/version9/{}/info/{:04d}.npz'.format(animal_name,frame+1)
        s_joint_dir = '{}/temparray/ske.npy'.format(source_path)

        ske_dict = np.load(s_joint_dir,allow_pickle=True).item()




        # cam_mat = np.loadtxt(source_cam_path)
        cam_mat = np.load(source_cam_path)['intrinsic_mat']
        source_focal = cam_mat[0,0]

        target_path = '/projects/perception/datasets/animal_videos/version9/{}/frame_{:06d}.obj'.format(animal_name,frame+1)

        target_info_path = '/projects/perception/datasets/animal_videos/version9/{}/info/{:04d}.npz'.format(animal_name,frame+1)
        gt_ske_all = \
            np.load(os.path.join('/projects/perception/datasets/animal_videos/version9', animal_name, 'skeleton', 'skeleton_all_frames.npy'), allow_pickle=True).item()

        gt_ske = gt_ske_all['frame_{:06d}'.format(frame + 1)]

        joint_list = []
        for key in gt_ske.keys():
            if key == 'def_c_root_joint':
                continue

            joint_list.append(((gt_ske[key]['head'] + gt_ske[key]['tail']) / 2)[np.newaxis, :])

        gt_joint_array = np.concatenate(joint_list, axis=0)

        t_info = np.load(target_info_path)

        s_mesh = trimesh.load(source_mesh_path,process=False)

        s_p,s_n,s_f = read_obj(source_mesh_path)
        s_mesh.vertices = s_p
        s_mesh.faces = s_f


        t_mesh = trimesh.load(target_path,process=False)

        # pdb.set_trace()
        t_mesh.vertices = t_mesh.vertices[:, [0, 2, 1]]
        t_mesh.vertices[:, 1] *= -1

        # if s_joint_dir[-4:] == '.npy':
        #     s_joint = np.load(s_joint_dir)
        # else:
        #     s_joint = trimesh.load(s_joint_dir,process=False).vertices

        s_joint = []
        for key in ske_dict.keys():
            if key=='def_c_root_joint':
                continue

            if 'joey' in key:
                continue
            # print(key)
            head = ske_dict[key]['new_head'][frame]
            tail = ske_dict[key]['new_tail'][frame]
            s_joint.append(((head + tail) / 2)[np.newaxis, :])

        # pdb.set_trace()
        s_joint_array = np.concatenate(s_joint, axis=0)

        # pdb.set_trace()

        source_joint_num = len(s_joint_array)
        s_mesh.vertices = np.concatenate((s_mesh.vertices, s_joint_array))

        gt_joint_num = len(gt_joint_array)
        t_mesh.vertices = np.concatenate((t_mesh.vertices,gt_joint_array))


        #### gt_mesh_coord
        # t_info = np.load(target_info_path)
        t_cam_rot = t_info['cam_rot']
        t_cam_loc = t_info['cam_loc']
        t_mesh.vertices = t_mesh.vertices @ t_cam_rot
        t_mesh.vertices[:,-1] += np.linalg.norm(t_cam_loc)

        target_focal = t_info['intrinsic_mat'][0,0]

        #### pred_mesh_coord
        # s_mesh.vertices[:,[1]] *= -1
        s_mesh.vertices = s_mesh.vertices[:, [0, 2, 1]]
        s_mesh.vertices[:, 1] *= -1

        s_mesh.vertices = s_mesh.vertices @ t_cam_rot
        s_mesh.vertices[:,-1] += np.linalg.norm(t_cam_loc)



        # pdb.set_trace()

        s_mean = np.mean(s_mesh.vertices,axis=0)
        s_mesh.vertices -= s_mean
        s_mesh.vertices *= source_focal/target_focal
        s_mesh.vertices += s_mean
        # s_mesh

        s_mesh_tmp = s_mesh.copy()
        t_mesh_tmp = t_mesh.copy()


        init_scale =  t_mesh_tmp.vertices[:,-1].mean()/s_mesh_tmp.vertices[:,-1].mean()

        s_mesh.vertices *= (init_scale)

        # pdb.set_trace()
        s_mesh.vertices[:, 0] -= s_mesh.vertices[:, 0].min()
        t_mesh.vertices[:, 0] -= t_mesh.vertices[:, 0].min()
        s_mesh.vertices[:, 1] -= s_mesh.vertices[:, 1].min()
        t_mesh.vertices[:, 1] -= t_mesh.vertices[:, 1].min()

        # pdb.set_trace()
        ### icp, trans is result pcd, mat is rotation matrix
        # mat, trans, cost = trimesh.registration.icp(s_mesh.vertices, t_mesh.vertices, max_iterations=1000)
        #
        # s_mesh.vertices = trans

        # s_mesh_tmp = np.concatenate((np.array(s_mesh.vertices), np.ones((s_mesh.vertices.shape[0],1)) ),axis=1)
        # rot_result = (mat@s_mesh_tmp.T).T

        factor_list = [0.95,0.96,0.97,0.98,0.99,1,1.01,1.02,1.03,1.04,1.05]

        # factor_list = [1]

        tmp_record = []
        for factor in factor_list:

            s_mesh_tmp = s_mesh.copy()
            t_mesh_tmp = t_mesh.copy()

            # s_mesh_tmp.vertices *= (init_scale*factor)
            s_mesh_tmp.vertices *= factor

            iou1 = iiou.compute_metric_obj(t_mesh_tmp, s_mesh_tmp,all_center=True)

            tmp_record.append(iou1)

        iou = max(tmp_record)
        factor_choose = factor_list[np.argmax(np.array(tmp_record))]
        s_mesh.vertices *= factor_choose

        # pdb.set_trace()

        s_joints = np.array(s_mesh.vertices)[-source_joint_num:, :]
        s_mesh.vertices = s_mesh.vertices[:-source_joint_num]

        t_joints = np.array(t_mesh.vertices)[-gt_joint_num:,:]
        t_mesh.vertices = t_mesh.vertices[:-gt_joint_num,:]

        # pdb.set_trace()


        bone_CD = cham_loss(torch.tensor(s_joints)[None,:].float().cuda(),
                            torch.tensor(t_joints)[None,:].float().cuda())[0].mean() + cham_loss(torch.tensor(s_joints)[None,:].float().cuda(),
                            torch.tensor(t_joints)[None,:].float().cuda())[1].mean()

        chamfer_loss = cham_loss(torch.tensor(np.array(s_mesh.vertices))[None, :].float().cuda(),
                  torch.tensor(np.array(t_mesh.vertices))[None, :].float().cuda())[0].mean() + cham_loss(torch.tensor(np.array(s_mesh.vertices))[None, :].float().cuda(),
                  torch.tensor(np.array(t_mesh.vertices))[None, :].float().cuda())[1].mean()

        if frame==0:
            min_cost = get_skin_cost(s_mesh,animal_name=animal_name)
            skin_list.append(min_cost)
        iou_list.append(iou)

        chamfer_list.append(chamfer_loss.detach().cpu().numpy())

        bone_cd_list.append(bone_CD.detach().cpu().numpy())

        sample_num += 1


    m_IOU = np.array(iou_list).mean()
    m_CD = np.array(chamfer_list).mean()
    m_BCD = np.array(bone_cd_list).mean()
    m_skin = np.array(skin_list).mean()

    print('Average IOU', m_IOU)
    print('Average Chamfer', m_CD)
    print('Average Bone chamfer ', m_BCD)
    print('Min Skin Chamfer ',m_skin)
    # np.save(os.path.join(out_dir, '{}.npy'.format(animal_name)), np.array(iou_list))
            # pdb.set_trace()

    return m_IOU,m_CD,m_BCD,m_skin



if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='instruct')
    parser.add_argument('--index',default=0,type=int)
    args = parser.parse_args()
    # animal_name = 'arctic_wolf_male'

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

    iou_record = []
    cd_record = []
    bcd_record = []
    skin_record = []
    for animal_name in val_split_list:
        # index = args.index
        # animal_name = val_split_list[index]
        m_iou, m_cd,m_bcd,m_skin = get_cam_IOU(animal_name)
        iou_record.append(m_iou)
        cd_record.append(m_cd)
        bcd_record.append(m_bcd)
        skin_record.append(m_skin)
    # for animal_name in val_split_list:
    #     icp_get_new(animal_name)

    print("MIOU ",np.array(iou_record).mean())
    print('MCD ',np.array(cd_record).mean())
    print("MBCD ",np.array(bcd_record).mean())
    print('MSKIN ',np.array(skin_record).mean())



















