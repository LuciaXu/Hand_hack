import re
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = '/media/data_cifs/lakshmi/LeapMotion/170616150626/'
finger_db = ['TYPE_THUMB','TYPE_INDEX','TYPE_MIDDLE','TYPE_RING','TYPE_PINKY']
bones_db = ['TYPE_TIP','TYPE_METACARPAL','TYPE_PROXIMAL','TYPE_INTERMEDIATE','TYPE_DISTAL']

def detecthand(str):
    tags = str.split(',')
    dict={}
    for tag in tags:
        dict[tag.split(':')[0].strip()] = int(tag.split(':')[1])
    if dict['hands'] == 0:
        return False
    else:
        return True

def parseJointLocations(strs):
    dict={}
    for finger in finger_db:
        dict[finger] = {}

    finger=''

    for str in strs:
        finger_found=False
        idx = str.find('Finger id')
        if idx != -1:
            # found a finger. keep track of the index
            m = re.search('TYPE_[A-Z]*',str)
            finger = m.group(0)
            finger_found = True

        idx = str.find('Bone')
        if (idx != -1) or (finger_found):
            # found a bone belonging to the last recorded finger
            mf = re.search('TYPE_[A-Z]*',str)
            bone = mf.group(0)
            if finger_found:
                bone = 'TYPE_TIP'

            m = re.findall('[\(][^\)]*', str)
            coords_start = m[0][1:]
            coords_start = np.asarray(coords_start.split(','),dtype=np.float32)

            coords_end = np.asarray([0,0,0],dtype=np.float32)
            if (not finger_found):
                coords_end = m[1][1:]
                coords_end = np.asarray(coords_end.split(','),dtype=np.float32)

            dict[finger].update({bone:[coords_start,coords_end]})

    return dict

def displayJoints(ax,joints):
    ax.clear()
    ax.set_xlim([-300,300])
    ax.set_ylim([-300, 300])
    ax.set_zlim([-300, 300])
    ax.invert_zaxis()

    for finger in finger_db:
        for bone in bones_db:
            ax.scatter(joints[finger][bone][0][0],joints[finger][bone][0][1],joints[finger][bone][0][2])
            if bone == 'TYPE_TIP':
                continue
            x = [joints[finger][bone][0][0],joints[finger][bone][1][0]]
            y = [joints[finger][bone][0][1], joints[finger][bone][1][1]]
            z = [joints[finger][bone][0][2], joints[finger][bone][1][2]]

            ax.plot(x,y,z)
    plt.pause(0.001)

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    idx = 0
    while True:
        print idx
        filename = '{}leap_{}.txt'.format(path,idx)
        if not os.path.exists(filename):
            plt.show()
            exit(0)

        leapinfo = open(filename,'r')
        hand_found = detecthand(leapinfo.readline())
        if hand_found:
            joints = parseJointLocations(leapinfo.readlines())
            displayJoints(ax,joints)
        idx+=1
    plt.show()

if __name__ == "__main__":
    main()