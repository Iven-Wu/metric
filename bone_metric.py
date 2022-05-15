import pdb

import numpy as np
import os

import trimesh

from tqdm import tqdm

import argparse

def chamfer_dist(pt1, pt2):
    pt1 = pt1[np.newaxis, :, :]
    pt2 = pt2[:, np.newaxis, :]
    dist = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=2))
    min_left = np.mean(np.min(dist, axis=0))
    min_right = np.mean(np.min(dist, axis=1))
    #print(min_left, min_right)
    return (min_left + min_right) / 2


def oneway_chamfer(pt_src, pt_dst):
    pt1 = pt_src[np.newaxis, :, :]
    pt2 = pt_dst[:, np.newaxis, :]
    dist = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=2))
    avg_dist = np.mean(np.min(dist, axis=0))
    return avg_dist


def joint2bone_chamfer_dist(skel1, skel2):
    # bone_sample_1 = sample_skel(skel1)
    # bone_sample_2 = sample_skel(skel2)

    bone_sample_1
    bone_sample_2
    # joint_1 = getJointArr(skel1)
    # joint_2 = getJointArr(skel2)

    joints1
    joints2
    dist1 = oneway_chamfer(joint_1, bone_sample_2)
    dist2 = oneway_chamfer(joint_2, bone_sample_1)
    return (dist1 + dist2) / 2

def read_gt_joints(animal_name,frame=0):
    fpath3 = '/projects/perception/datasets/animal_videos/version9'
    ske = np.load(os.path.join(fpath3, animal_name,'skeleton', 'skeleton_all_frames.npy'), allow_pickle=True).item()[
        'frame_{:06d}'.format(frame+1)]

    info_path = '/projects/perception/datasets/animal_videos/version9/{}/info/{:04d}.npz'.format(animal_name, frame + 1)
    gt_info = np.load(info_path)
    gt_cam_rot = gt_info['cam_rot']
    gt_cam_loc = gt_info['cam_loc']


    for key in ske.keys():
        head, tail = ske[key]['head'], ske[key]['tail']
        head[[1]] *= -1
        tail[[1]] *= -1
        head, tail = head[[0, 2, 1]], tail[[0, 2, 1]]
        ske[key]['head'] = head
        ske[key]['tail'] = tail


    joint_list = []
    for key in ske.keys():
        if key=='def_c_root_joint':
            continue

        joint_list.append(((ske[key]['head']+ske[key]['tail'])/2)[np.newaxis,:])

    joint_array = np.concatenate(joint_list, axis=0)

    # pdb.set_trace()

    joint_array_new = joint_array @ gt_cam_rot
    joint_array_new[:,-1] += np.linalg.norm(gt_cam_loc)

    return joint_array_new

def read_lasr_joints(animal_name,frame=0):
    s_mesh_dir = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/pred{}.ply'.format(animal_name,frame)
    s_mesh = trimesh.load(s_mesh_dir)
    joint_dir = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/cam_bone{}.ply'.format(animal_name,frame)
    joint_location = np.array(trimesh.load(joint_dir).vertices)
    joint_location[:,1] *= -1

    source_cam_path = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/cam{}.txt'.format(animal_name, frame)
    cam_mat = np.loadtxt(source_cam_path)
    source_focal = cam_mat[-1, 0]


    pdb.set_trace()
    s_mesh.vertices = np.concatenate(s_mesh.vertices,joint_location)


    gt_info_path = '/home/yuefanw/scratch/version9/{}/info/{:04d}.npz'.format(animal_name,frame)
    gt_info = np.load(gt_info_path)
    # gt_mesh =
    target_focal = gt_info['intrinsic_mat'][0, 0]


    matrix = np.eye(4)
    matrix[:3, :3] *= target_focal / source_focal
    s_mesh.apply_transform(matrix)
    s_mesh.vertices[:, 1] -= s_mesh.vertices[:, 1].min()

    init_scale = t_mesh.vertices[:, -1].mean() / s_mesh_tmp.vertices[:, -1].mean()

    return joint_location

class lasr_CD_B2B():
    def __init__(self,animal_name='aardvark_female',frame=0):
        self.animal_name = animal_name
        self.frame = frame
        # s_mesh_dir = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/pred{}.ply'.format(animal_name, frame)
        # s_cam_dir =
        # s_cam_dir =

    def chamfer_dist(self,pt1, pt2):
        pt1 = pt1[np.newaxis, :, :]
        pt2 = pt2[:, np.newaxis, :]
        dist = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=2))
        min_left = np.mean(np.min(dist, axis=0))
        min_right = np.mean(np.min(dist, axis=1))
        # print(min_left, min_right)
        return (min_left + min_right) / 2

    def oneway_chamfer(self,pt_src, pt_dst):
        pt1 = pt_src[np.newaxis, :, :]
        pt2 = pt_dst[:, np.newaxis, :]
        dist = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=2))
        avg_dist = np.mean(np.min(dist, axis=0))
        return avg_dist

    def read_gt_joints(self):
        fpath3 = '/projects/perception/datasets/animal_videos/version9'
        ske_all = \
        np.load(os.path.join(fpath3, self.animal_name, 'skeleton', 'skeleton_all_frames.npy'), allow_pickle=True).item()

        ske = ske_all['frame_{:06d}'.format(self.frame + 1)]
        info_path = '/projects/perception/datasets/animal_videos/version9/{}/info/{:04d}.npz'.format(self.animal_name,
                                                                                                     self.frame + 1)
        gt_info = np.load(info_path)
        gt_cam_rot = gt_info['cam_rot']
        gt_cam_loc = gt_info['cam_loc']

        # for key in ske.keys():
        #     head, tail = ske[key]['head'], ske[key]['tail']
        #     head[[1]] *= -1
        #     tail[[1]] *= -1
        #     head, tail = head[[0, 2, 1]], tail[[0, 2, 1]]
        #     ske[key]['head'] = head
        #     ske[key]['tail'] = tail

        # pdb.set_trace()
        joint_list = []
        for key in ske.keys():
            if key == 'def_c_root_joint':
                continue

            joint_list.append(((ske[key]['head'] + ske[key]['tail']) / 2)[np.newaxis, :])

        # pdb.set_trace()
        joint_array = np.concatenate(joint_list, axis=0)

        joint_array_new = joint_array @ gt_cam_rot
        joint_array_new[:, -1] += np.linalg.norm(gt_cam_loc)

        joint_array_new[:,1] -= joint_array_new[:,1].min()

        return joint_array_new

    def get_gen_joints(self):

        ### for gen
        # s_mesh_path = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/pred{}.ply'.format(self.animal_name,self.frame)
        # s_cam_path = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/cam{}.txt'.format(self.animal_name, self.frame)
        # s_joint_dir = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/cam_bone{}.ply'.format(self.animal_name, self.frame)

        s_mesh_path = '/home/yuefanw/scratch/viser-release/log/{}-6/save/{}-vp1pred{}.obj'.format(self.animal_name,self.animal_name,self.frame)
        s_cam_path = '/home/yuefanw/scratch/viser-release/log/{}-6/save/{}-cam{}.txt'.format(self.animal_name, self.animal_name,self.frame)
        s_joint_dir = '/home/yuefanw/scratch/viser-release/log/{}-6/save/{}-bones{}.npy'.format(self.animal_name,self.animal_name, self.frame)


        s_mesh = trimesh.load(s_mesh_path)
        s_cam = np.loadtxt(s_cam_path)
        if s_joint_dir[-4:] =='.npy':
            s_joint = np.load(s_joint_dir)
        else:
            s_joint = trimesh.load(s_joint_dir).vertices

        source_focal = s_cam[-1, 0]
        joint_num = len(s_joint)

        s_mesh.vertices = np.concatenate((s_mesh.vertices,s_joint))
        s_mesh.vertices[:, [1]] *= -1


        ##### for gt
        t_mesh_path = '/home/yuefanw/scratch/version9/{}/frame_{:06d}.obj'.format(self.animal_name, self.frame + 1)
        t_info_path = '/home/yuefanw/scratch/version9/{}/info/{:04d}.npz'.format(self.animal_name, self.frame + 1)
        t_info = np.load(t_info_path)
        t_mesh = trimesh.load(t_mesh_path)

        t_mesh.vertices = t_mesh.vertices[:,[0,2,1]]
        t_mesh.vertices[:,1] *= -1

        t_cam_rot = t_info['cam_rot']
        t_cam_loc = t_info['cam_loc']
        t_mesh.vertices = t_mesh.vertices @ t_cam_rot
        t_mesh.vertices[:,-1] += np.linalg.norm(t_cam_loc)

        target_focal = t_info['intrinsic_mat'][0, 0]



        ### changing scale
        matrix = np.eye(4)
        matrix[:3, :3] *= target_focal/source_focal
        s_mesh.apply_transform(matrix)

        s_mesh.vertices[:,1] -= s_mesh.vertices[:,1].min()
        t_mesh.vertices[:,1] -= t_mesh.vertices[:,1].min()

        init_scale = t_mesh.vertices[:, -1].mean() / s_mesh.vertices[:, -1].mean()

        s_mesh.vertices *= (init_scale)

        # pdb.set_trace()
        s_joints = np.array(s_mesh.vertices)[-joint_num:,:]

        return s_joints

    def calculate_dis_1f(self):
        gt_joints = self.read_gt_joints()
        gen_joints = self.get_gen_joints()

        # pdb.set_trace()
        B2B_CD = self.chamfer_dist(gt_joints,gen_joints)

        return B2B_CD

    def calculate_dis_all(self):
        CD_list = []
        for i in tqdm(range(30)):
            self.frame = i
            cd = self.calculate_dis_1f()
            CD_list.append(cd)

        print('Avg CD ',np.array(CD_list).mean())

        return np.array(CD_list).mean()

#### joint to joint chamfer distance, input shape (m,3) (n,3)
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

    bone_CD = []
    for animal in val_split_list:
        print("For animal ",animal)
        cal = lasr_CD_B2B(animal_name=animal)
        BCD = cal.calculate_dis_all()
        bone_CD.append(BCD)

    print("AVG all ",np.array(bone_CD).mean())

    pdb.set_trace()
    # joint_array = read_gt_joints('aardvark_female')
    #
    # pdb.set_trace()
    #
    # joint_array2 = read_lasr_joints('aardvark_female')
    # pdb.set_trace()
    # J2J_CD = chamfer_dist(joints1,joints2)
    #
    #
    # #### Bone to bone chamfer distance
    # B2B_CD = chamfer_dist(bones1,bones2)
    #
    # ### Joint to bone chamfer distance
    # B2J_CD = chamfer_dist()