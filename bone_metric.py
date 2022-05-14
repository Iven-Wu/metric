import pdb

import numpy as np
import os

import trimesh

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
    return joint_array
    # if np.linalg.norm(ske[key]['head'] - ske[ske[key]['parent']]['tail']) > 1e-6:
    #     ske[key]['head'],ske[key]['tail'] = ske[key]['tail'] , ske[key]['head']
    # print(key)
    # print("Distacne ",np.linalg.norm(ske[key]['head']-ske[ske[key]['parent']]['tail']))
    # print("Distance_new ", np.linalg.norm(ske[key]['head'] - ske[ske[key]['parent']]['head']))
    # joints = []
    # this_level = [skel.root]
    # while this_level:
    #     next_level = []
    #     for p_node in this_level:
    #         joint_ = np.array(p_node.pos)
    #         joint_ = joint_[np.newaxis, :]
    #         joints.append(joint_)
    #         next_level += p_node.children
    #     this_level = next_level
    # joints = np.concatenate(joints, axis=0)

    # pdb.set_trace()
    # min_distance = np.min([np.linalg.norm(ske[key]['head']-ske[ske[key]['parent']]['head']),
    #                    np.linalg.norm(ske[key]['head']-ske[ske[key]['parent']]['tail']),
    #                    np.linalg.norm(ske[key]['tail']-ske[ske[key]['parent']]['head']),
    #                    np.linalg.norm(ske[key]['tail']-ske[ske[key]['parent']]['tail'])])
    # print(min_distance)

    # joint_list = []
    #
    # for key in ske.keys():
    #     if key =='def_c_root_joint':
    #         continue

    # pdb.set_trace()

def read_lasr_joints(animal_name,frame=0):
    joint_dir = '/home/yuefanw/scratch/lasr/log/{}-5/save/cam_bone{}.ply'.format(animal_name,frame)
    joint_location = np.array(trimesh.load(joint_dir).vertices)

    return joint_location


#### joint to joint chamfer distance, input shape (m,3) (n,3)
if __name__ =='__main__':

    joint_array = read_gt_joints('aardvark_female')

    joint_array2 = read_lasr_joints('aardvark_female')
    pdb.set_trace()
    # J2J_CD = chamfer_dist(joints1,joints2)
    #
    #
    # #### Bone to bone chamfer distance
    # B2B_CD = chamfer_dist(bones1,bones2)
    #
    # ### Joint to bone chamfer distance
    # B2J_CD = chamfer_dist()