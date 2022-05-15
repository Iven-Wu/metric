import pdb

import numpy as np
import os

import trimesh

from tqdm import tqdm

import argparse


import scipy

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

class lasr_skin():
    def __init__(self,animal_name='aardvark_female',frame=0):
        self.animal_name = animal_name
        self.frame = frame


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

    def get_points(self):

        ### for gen
        self.source_root = '/home/yuefanw/scratch/lasr/raw_log/{}-5/save/'.format(self.animal_name)
        s_mesh_path = '{}/pred{}.ply'.format(self.source_root,self.frame)
        s_cam_path = '{}/cam{}.txt'.format(self.source_root, self.frame)
        s_joint_dir = '{}/cam_bone{}.ply'.format(self.source_root, self.frame)

        # self.source_root = '/home/yuefanw/scratch/viser-release/log/{}-6/save'.format(self.animal_name)
        #
        # s_mesh_path = '{}/{}-vp1pred{}.obj'.format(self.source_root,self.animal_name,self.frame)
        # s_cam_path = '{}/{}-cam{}.txt'.format(self.source_root, self.animal_name,self.frame)
        # s_joint_dir = '{}/{}-bones{}.npy'.format(self.source_root,self.animal_name, self.frame)

        s_mesh = trimesh.load(s_mesh_path)
        s_cam = np.loadtxt(s_cam_path)
        if s_joint_dir[-4:] =='.npy':
            s_joint = np.load(s_joint_dir)
        else:
            s_joint = trimesh.load(s_joint_dir).vertices

        source_focal = s_cam[-1, 0]
        joint_num = len(s_joint)

        # s_mesh.vertices = np.concatenate((s_mesh.vertices,s_joint))
        s_mesh.vertices[:, [1]] *= -1


        ##### for gt
        # t_mesh_path = '/home/yuefanw/scratch/version9/{}/frame_{:06d}.obj'.format(self.animal_name, self.frame + 1)

        self.target_root = '/home/yuefanw/scratch/simplified_meshes/simplified_meshes/{}'.format(self.animal_name)
        t_mesh_path = '{}/remesh.obj'.format(self.target_root)
        t_info_path = '/home/yuefanw/scratch/version9/{}/info/{:04d}.npz'.format(self.animal_name, self.frame + 1)
        t_info = np.load(t_info_path)

        t_mesh = trimesh.load(t_mesh_path)
        points,normals,faces = read_obj(t_mesh_path)
        # pdb.set_trace()
        t_mesh.vertices = points
        t_mesh.faces = faces

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
        # s_joints = np.array(s_mesh.vertices)[-joint_num:,:]

        return np.array(s_mesh.vertices) , np.array(t_mesh.vertices)

    def get_cost_matrix(self,s_mesh_p,t_mesh_p):
        s_weight = np.load(os.path.join(self.source_root,'skin.npy'))
        t_weight = np.load(os.path.join(self.target_root,'W1.npy')).T

        # if t_weight[-1].sum() < 0.1:
        #     t_weight
        while t_weight[-1].sum() < 0.1:
            t_weight = t_weight[:-1]

        # pdb.set_trace()

        cost_matrix = np.zeros((s_weight.shape[0],t_weight.shape[0]))
        for s_bone in range(s_weight.shape[0]):
            # s_weight[]
            # pdb.set_trace()
            s_index = np.where(s_weight[s_bone]>0.01)[0]
            s_points_loc = s_mesh_p[s_index]

            for t_bone in range(t_weight.shape[0]):
                t_index = np.where(t_weight[t_bone]>0.01)[0]
                t_points_loc = t_mesh_p[t_index]

                # pdb.set_trace()
                # print('s_points_loc',s_points_loc.shape)
                # print('t_points_loc',t_points_loc.shape)
                # pdb.set_trace()

                if t_points_loc.shape[0]==0:
                    cost_matrix[s_bone,t_bone] = 100
                else:
                    cost_matrix[s_bone,t_bone] = self.chamfer_dist(s_points_loc,t_points_loc)

                # pdb.set_trace()

        row_ind,col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        min_cost = cost_matrix[row_ind,col_ind].sum()

        # pdb.set_trace()
        return min_cost

    def calculate_skin_distance(self):
        # gt_joints = self.read_gt_joints()
        # gen_joints = self.get_gen_points()

        s_mesh_p,t_mesh_p = self.get_points()

        # pdb.set_trace()
        skin_distance = self.get_cost_matrix(s_mesh_p,t_mesh_p)

        # pdb.set_trace()
        # B2B_CD = self.chamfer_dist(gt_joints,gen_joints)

        return skin_distance

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

    for animal in val_split_list:
        print("For animal ",animal)
        cal = lasr_skin(animal_name=animal)
        skin_dis = cal.calculate_skin_distance()

        print(skin_dis)
    # bone_CD = []
    # for animal in val_split_list:
    #     print("For animal ",animal)
    #     cal = lasr_CD_B2B(animal_name=animal)
    #     BCD = cal.calculate_dis_all()
    #     bone_CD.append(BCD)

    # print("AVG all ",np.array(bone_CD).mean())

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