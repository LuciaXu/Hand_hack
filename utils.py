'''
A set of utility functions for manipulation of annotated hand data
'''
import numpy as np
import cv2

'''
These are camera parameter for the NYU Hand dataset.
TODO: Import these values from the dataset class
'''
fx = 588.03
fy = 587.07
ux = 320.
uy = 240.

def jointsImgTo3D(sample):
    '''
    U-V-D coordinate system to X-Y-Z coordinate system
    :param sample: uvd of a set of joints
    :return: xyz of the corresponding joints
    '''
    ret = np.zeros((sample.shape[0], 3), np.float32)
    for i in range(sample.shape[0]):
        ret[i] = jointImgTo3D(sample[i])
    return ret


def jointImgTo3D(sample):
    '''
    U-V-D coordinate system to X-Y-Z coordinate system
    :param sample: uvd of a single joint
    :return: xyz of the joint
    '''
    ret = np.zeros((3,), np.float32)
    # convert to metric using f, see Thomson et al.
    ret[0] = (sample[0] - ux) * sample[2] / fx
    ret[1] = (uy - sample[1]) * sample[2] / fy
    ret[2] = sample[2]
    return ret


def joints3DToImg(sample):
    '''
    X-Y-Z coordinate system to U-V-D coordinate system
    :param sample: xyz of a group of joints
    :return: uvd of the corresponding joints
    '''
    ret = np.zeros((sample.shape[0], 3), np.float32)
    for i in range(sample.shape[0]):
        ret[i] = joint3DToImg(sample[i])
    return ret


def joint3DToImg(sample):
    '''
    X-Y-Z coordinate system to U-V-D coordinate system
    :param sample: xyz of a single joint
    :return: uvd of the specified joint
    '''
    ret = np.zeros((3,), np.float32)
    # convert to metric using f, see Thomson et.al.
    if sample[2] == 0.:
        ret[0] = ux
        ret[1] = uy
        return ret
    ret[0] = sample[0] / sample[2] * fx + ux
    ret[1] = uy - sample[1] / sample[2] * fy
    ret[2] = sample[2]
    return ret

def rotatePoint2D(p1, center, angle):
        """
        Rotate a point in 2D around center
        :param p1: point in 2D (u,v,d)
        :param center: 2D center of rotation
        :param angle: angle in deg
        :return: rotated point
        """
        alpha = angle * np.pi / 180.
        pp = p1.copy()
        pp[0:2] -= center[0:2]
        pr = np.zeros_like(pp)
        pr[0] = pp[0] * np.cos(alpha) - pp[1] * np.sin(alpha)
        pr[1] = pp[0] * np.sin(alpha) + pp[1] * np.cos(alpha)
        pr[2] = pp[2]
        ps = pr
        ps[0:2] += center[0:2]
        return ps

'''
TODO: Cube should be a parameter
currenty we assume cube to be [300,300,300]
'''
def rotateHand(image, com, joints3D, dims):
    """
    Please do note that this function is different from the namesake in handdetector.py!
    Rotate hand virtually in the image plane by a given angle
    :param com: original center of mass, in **3D** coordinates (x,y,z)
    :param rot: rotation angle in deg
    :param joints3D: original joint coordinates, in 3D coordinates (x,y,z)
    :param dims: dimensions of the image
    :return: new 3D joint coordinates
    """
    rot = np.random.uniform(0, 360)

    cubez = 300
    # rescale joints
    joints3D = joints3D*(cubez/2.0)
    # get the uvd coordinates of the center of mass
    comUVD = joint3DToImg(com)
    # if rot is 0, nothing to do
    if np.allclose(rot, 0.):
        joints3D = np.clip(np.asarray(joints3D, dtype='float32') / (cubez/2.0), -1, 1)
        return joints3D

    # For a non-zero rotation!
    rot = np.mod(rot, 360)
    # get the 2D rotation matrix
    M = cv2.getRotationMatrix2D((dims[1] // 2, dims[0] // 2), -rot, 1)

    image = cv2.warpAffine(image, M, (dims[1], dims[0]), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=1)

    # translate to COM and project on to the image
    joint_2D = joints3DToImg(joints3D + com)
    # rotate every joint in plane
    data_2D = np.zeros_like(joint_2D)
    for k in xrange(data_2D.shape[0]):
        data_2D[k] = rotatePoint2D(joint_2D[k], comUVD[0:2], rot)
    # inverse translate
    new_joints3D = (jointsImgTo3D(data_2D) - com)
    # clip the limits of the joints
    new_joints3D = np.clip(np.asarray(new_joints3D, dtype='float32') / (cubez / 2.0), -1, 1)

    return image,new_joints3D

def augment_sample(label,image,com3D,M,dims):
    # possible augmentation modes
    aug_modes = ['rot', 'scale', 'trans']
    # pick an augmentation method
    #mode = np.random.randint(0, len(aug_modes))
    mode = 0

    if aug_modes[mode] == 'rot':
        image, label = rotateHand(image, com3D, label, dims)

    '''
    elif aug_modes[mode] == 'scale':
        imgD, new_joints3D, cube, M = hd.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M)
    elif aug_modes[mode] == 'trans':
        imgD, new_joints3D, com, M = hd.transHand(img.astype('float32'), cube, com, off, gt3Dcrop, M)
    else:
        print('Such an augmentation method has not be implemented')
    '''
    return label,image,com3D,M