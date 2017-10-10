import re
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D

path_good = '/media/playroom_data/3697/LeapMotion/170315154638/'

#path_good = '/media/playroom_data/3735/LeapMotion/170327162917/'
basepath = '/media/playroom_data/'

#path = '/media/data_cifs/lakshmi/LeapMotion/170907173018/'

finger_db = ['TYPE_THUMB','TYPE_INDEX','TYPE_MIDDLE','TYPE_RING','TYPE_PINKY']
bones_db = ['TYPE_TIP','TYPE_METACARPAL','TYPE_PROXIMAL','TYPE_INTERMEDIATE','TYPE_DISTAL']

def detecthand(str):
    if len(str) == 0:
        return False

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
        #print len(str)
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
    ax.grid('off')
    ax.set_xlim([-300,300])
    ax.set_ylim([-300, 300])
    ax.set_zlim([-300, 300])
    ax.invert_zaxis()

    for finger in finger_db:
        for bone in bones_db:
            if bone == 'TYPE_TIP':
                continue
            ax.scatter(joints[finger][bone][0][0],joints[finger][bone][0][1],joints[finger][bone][0][2],color='gray',alpha=0.5)
            x = [joints[finger][bone][0][0],joints[finger][bone][1][0]]
            y = [joints[finger][bone][0][1], joints[finger][bone][1][1]]
            z = [joints[finger][bone][0][2], joints[finger][bone][1][2]]

            c = 'gray'
            a = 0.4
            if finger == 'TYPE_THUMB' or finger == 'TYPE_INDEX':
                c = 'red'
                a = 1.0
            ax.plot(x,y,z,color=c,alpha=a)
    plt.pause(0.001)
    #plt.show()

def processSubject(path,disp=False):
    nTrails = 0
    freshDetect = True

    if disp:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

    idx = 0
    gripaperture = []

    while True:
        print idx
        filename = '{}leap_{}.txt'.format(path,idx)
        if not os.path.exists(filename):
            break

        leapinfo = open(filename,'r')
        hand_found = detecthand(leapinfo.readline())
        if hand_found:
            if freshDetect == True:
                nTrails=nTrails+1
                freshDetect = False

            joints = parseJointLocations(leapinfo.readlines())
            diff = np.asarray(joints['TYPE_THUMB']['TYPE_TIP'][0] - joints['TYPE_INDEX']['TYPE_TIP'][0])
            dist = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
            gripaperture.append(dist)
            if disp:
                displayJoints(ax,joints)
        else:
            gripaperture.append(0)
            freshDetect = True

        idx+=1

    if disp:
        plt.show()
    print('Number of trails: %d'%nTrails)
    return nTrails, gripaperture

def smoothListGaussian(list,strippedXs=False,degree=15):

     window=degree*2-1
     weight=np.array([1.0]*window)
     weightGauss=[]

     for i in range(window):
         i=i-degree+1
         frac=i/float(window)
         gauss=1/(np.exp((4*(frac))**2))
         weightGauss.append(gauss)

     weight=np.array(weightGauss)*weight
     smoothed=[0.0]*(len(list)-window)

     for i in range(len(smoothed)):
         smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)
     return smoothed

def main():

    #processSubject(path_good,disp=True)

    t = 0
    trails = []
    folders = glob.glob(basepath+'[0-9]*')
    for subject in folders:
        tmp_path = subject+'/LeapMotion/'
        path = glob.glob(tmp_path+'[0-9]*')
        if len(path) == 0:
            continue
        print path
        ntrails,gripaperture = processSubject(path[0]+'/')
        trails.append(ntrails)

        if not (ntrails == 0):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(gripaperture,color='red',alpha=0.5)
            ax.plot(smoothListGaussian(gripaperture))
            ax.set_xlabel('Time')
            ax.set_ylabel('Grip Aperture')
            ax.set_title(path)
            plt.savefig(str(t)+'.png')
            plt.close()
            t=t+1

    plt.figure()
    plt.hist(trails)
    plt.show()

if __name__ == "__main__":
    main()