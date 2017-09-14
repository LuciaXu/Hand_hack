import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image

def joints3DToImg(sample):
    ret = np.zeros((sample.shape[0], 3), np.float32)
    for i in range(sample.shape[0]):
        ret[i] = joint3DToImg(sample[i])
    return ret

def joint3DToImg(sample):
    ux = 315.944855
    #ux = 320
    uy = 245.287079
    #uy = 240
    fx = 475.065948
    fy = 475.06585
    ret = np.zeros((3,), np.float32)
    # convert to metric using f, see Thomson et.al.
    if sample[2] == 0.:
        ret[0] = ux
        ret[1] = uy
        return ret
    ret[0] = sample[0] / sample[2] * fx + ux
    ret[1] = sample[1] / sample[2] * fy+uy
    ret[2] = sample[2]
    return ret

def showImgJoints(imageposition,joint2d,line=True):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    img = Image.imread(imageposition)
    ax.imshow(img)
    ax.scatter(joint2d[:, 0], joint2d[:, 1], color='r')
    if line==True:
        ax.plot([joint2d[0,0],joint2d[1,0]],[joint2d[0,1],joint2d[1,1]],c='m')
        ax.plot([joint2d[0, 0], joint2d[2, 0]], [joint2d[0, 1], joint2d[2, 1]], c='b')
        ax.plot([joint2d[0, 0], joint2d[3, 0]], [joint2d[0, 1], joint2d[3, 1]], c='g')
        ax.plot([joint2d[0, 0], joint2d[4, 0]], [joint2d[0, 1], joint2d[4, 1]], c='y')
        ax.plot([joint2d[0, 0], joint2d[5, 0]], [joint2d[0, 1], joint2d[5, 1]], c='r')

        ax.plot([joint2d[1,0],joint2d[6,0]], [joint2d[1, 1], joint2d[6, 1]], c='m')
        ax.plot([joint2d[6, 0], joint2d[7, 0]], [joint2d[6, 1], joint2d[7, 1]], c='m')
        ax.plot([joint2d[7, 0], joint2d[8, 0]], [joint2d[7, 1], joint2d[8, 1]], c='m')

        ax.plot([joint2d[2, 0], joint2d[9, 0]], [joint2d[2, 1], joint2d[9, 1]], c='b')
        ax.plot([joint2d[9, 0], joint2d[10, 0]], [joint2d[9, 1], joint2d[10, 1]], c='b')
        ax.plot([joint2d[10, 0], joint2d[11, 0]], [joint2d[10, 1], joint2d[11, 1]], c='b')

        ax.plot([joint2d[3, 0], joint2d[12, 0]], [joint2d[3, 1], joint2d[12, 1]], c='g')
        ax.plot([joint2d[12, 0], joint2d[13, 0]], [joint2d[12, 1], joint2d[13, 1]], c='g')
        ax.plot([joint2d[13, 0], joint2d[14, 0]], [joint2d[13, 1], joint2d[14, 1]], c='g')

        ax.plot([joint2d[4, 0], joint2d[15, 0]], [joint2d[4, 1], joint2d[15, 1]], c='y')
        ax.plot([joint2d[15, 0], joint2d[16, 0]], [joint2d[15, 1], joint2d[16, 1]], c='y')
        ax.plot([joint2d[16, 0], joint2d[17, 0]], [joint2d[16, 1], joint2d[17, 1]], c='y')

        ax.plot([joint2d[5, 0], joint2d[18, 0]], [joint2d[5, 1], joint2d[18, 1]], c='r')
        ax.plot([joint2d[18, 0], joint2d[19, 0]], [joint2d[18, 1], joint2d[19, 1]], c='r')
        ax.plot([joint2d[19, 0], joint2d[20, 0]], [joint2d[19, 1], joint2d[20, 1]], c='r')
    plt.show()

trainlablels='/media/data_cifs/lu/Challenge/data/Training_Annotation.txt'
e =0
with open(trainlablels) as f:
    for line in f:
        if (e==0):
            l=line.split('\t')
            imagename=l[0]
            jj=1
            joint=np.zeros([21,3],np.float32)

            for j in range(0,21):
                joint[j,0]=float(l[jj])
                jj+=1
                joint[j, 1] = float(l[jj])
                jj += 1
                joint[j, 2] = float(l[jj])
                jj += 1
            joint2d = joints3DToImg(joint)
            imageposition='/media/data_cifs/lu/Challenge/data/training/images/{}'.format(imagename)
            showImgJoints(imageposition,joint2d)
        e=e+1


