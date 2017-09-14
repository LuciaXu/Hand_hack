import os
import progressbar as pb
import cPickle
import numpy as np
from basetype import ICVLFrame, NamedImgSequence
from handdetector import HandDetector
from transformations import transformPoint2D
import scipy.io
#from PIL import Image
import matplotlib.image as Image
from check_fun import showAnnotatedDepth, showdepth,showImageLable,trans3DToImg,trans3DsToImg


class Importer(object):
    def __init__(self,fx,fy,ux,uy):
        """""
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """""
        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f, see Thomson et al.
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret
    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,),np.float32)
        #convert to metric using f, see Thomson et.al.
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = self.uy-sample[1]/sample[2]*self.fy
        ret[2] = sample[2]
        return ret


class ICVLImporter(Importer):

    def __init__(self, path,useCache=True, cacheDir = './cache/'):

        super(ICVLImporter, self).__init__(241.42,241.42,160.,120.)

        self.path = path
        self.numJoints = 16
        self.useCache = useCache
        self.cacheDir = cacheDir

    def test(self):
        lablefile = self.path + "/Training/labels.txt"
        print(lablefile)
        print(os.path.abspath(lablefile))
        os.listdir(self.path)
        if not os.path.isfile(lablefile):
            print("not existed")

    def loadSequence(self, seqname, subseq = None, Nmax = float('inf'),docom = False,shuffle = False,rng = None):
        """""
        load an image sequence from the dataset
        :param seqname: sequence name
        :param subseq: list of subsequence names
        :param Nmax: maximum number of samples to load
        :return: returns named image sequence
        """""
        if(subseq is not None) and (not isinstance(subseq, list)):
            raise TypeError("Subseq must be None or list")

        config = {'cube': (250, 250, 250)}
        refineNet = None

        if subseq is None:
            pickleCache = '{}/{}_{}_None_{}_cache.pkl'.format(self.cacheDir,self.__class__.__name__,seqname,docom)
        else:
            pickleCache = '{}/{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqname,''.join(subseq), docom)
        if self.useCache:
            if os.path.isfile(pickleCache):
                print('Loading cache data from {}'.format(pickleCache))
                f = open(pickleCache,'rb')
                (seqname,data,config) = cPickle.load(f)
                f.close()

                #shuffle data
                if shuffle and rng is not None:
                    print("shuffling")
                    rng.shuffle(data)
                if not(np.isinf(Nmax)):
                    return NamedImgSequence(seqname,data[0:Nmax],config)
                else:
                    return NamedImgSequence(seqname,data,config)

            #check for multiple subsequences
            if subseq is not None:
                if len(subseq) > 1:
                    missing = False
                    for i in range(len(subseq)):
                        if not os.path.isfile('{}/{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqname, subseq[i], docom)):
                            missing = True
                            print("missing:{}".format(subseq[i]))
                            break

                    if not missing:
                        #load first data
                        pickleCache = '{}/{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqname, subseq[0], docom)
                        print("Loading cache data from {}".format(pickleCache))
                        f = open(pickleCache,'rb')
                        (seqname,fullData,config) = cPickle.load(f)
                        f.close()
                        #load rest of data
                        for i in range(1,len(subseq)):
                            pickleCache = '{}/{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__,
                                                                            seqname, subseq[i], docom)
                            print("Loading cache data from {}".format(pickleCache))
                            f = open(pickleCache, 'rb')
                            (seqName, data, config) = cPickle.load(f)
                            fullData.extend(data)
                            f.close()

                        #shuffle data
                        if shuffle and rng is not None:
                            print("shuffling")
                            rng.shuffle(fullData)
                        if not(np.isinf(Nmax)):
                            return NamedImgSequence(seqname,fullData[0:Nmax],config)
                        else:
                            return NamedImgSequence(seqname,fullData,config)
        #load the dataset
        objdir = '{}/Training/Depth'.format(self.path)
        trainlabels = '{}/Training/{}.txt'.format(self.path,seqname)
        print('the trainlable is %s'%trainlabels)

        inputfile = open(trainlabels)

        txt = 'Loading {}'.format(seqname)
        pbar = pb.ProgressBar(maxval=len(inputfile.readlines()), widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()
        inputfile.seek(0)

        data = []
        i=0
        for line in inputfile:
            part = line.split('/')
            print part[0]
            #check of subsequences and skip them if necessary
            subseqname = ''
            if subseq is not None:
                p = part[0].split('/')
                #handle original data (unrotated '0') separately
                if ('0' in subseq) and len(p[0])>6:
                    pass
                elif not('0' in subseq) and len(p[0])>6:
                    i+=1
                    continue
                elif (p[0] in subseq) and len(p[0])<=6:
                    pass
                elif not(p[0] in subseq) and len(p[0])<=6:
                    i+=1
                    continue

                if len(p[0])<=6:
                    subseqname = p[0]
                else:
                    subseqname = '0'

            dptFileName = '{}/{}'.format(objdir, part[0])

            if not os.path.isfile(dptFileName):
                print("File {} dose not exist!".format(dptFileName))
                i+=1
                continue
            dpt = self.loadSequence(dptFileName)

            #joints in image coordinates
            gtorig = np.zeros((self.numJoints,3),np.float23)
            for joint in range(self.numJoints):
                for xyz in range(0,3):
                    gtorig[joint,xyz] = part[joint*3+xyz+1]

            #normalized joints in 3D coordinates
            gt3Dorig = self.jointsImgTo3D(gtorig)


            #Detect hand
            hd = HandDetector(dpt, self.fx, self.fy, refineNet=None, importer=self)
            if not hd.checkImage(1):
                print("Skipping image {}, no content".format(dptFileName))
                i+=1
                continue
            try:
                dpt, M, com = hd.cropArea3D(gtorig[0],size = config['cube'],docom = True)
            except UserWarning:
                print("Skipping image {}, no hand detected".format(dptFileName))
                continue

            com3D = self.jointImgTo3D(com)
            gt3Dcrop = gt3Dorig - com3D #normalize to com
            gtcrop = np.zeros((gtorig.shape[0],3),np.float32)
            for joint in range(gtorig.shape[0]):
                t = transformPoint2D(gtorig[joint],M)
                gtcrop[joint,0] = t[0]
                gtcrop[joint,1] = t[1]
                gtcrop[joint,2] = gtorig[joint,2]

            print("{}".format(gt3Dorig))

            data.append(ICVLFrame(dpt.astype(np.float32),gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,subseqname) )
            pbar.update(i)
            i+=1

            #early stop
            if len(data)>= Nmax:
                break

        inputfile.close()
        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save chache data to {}".format(pickleCache))
            f = open(pickleCache,'wb')
            cPickle.dump((subseqname,data,config),f,protocal = cPickle.HIGHEST_PROTOCOL)
            f.close()

        #shuffle data
        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)
        return  NamedImgSequence(seqname,data,config)


class ICCVChallengeImporter(Importer):
    def __init__(self,path,useCache = True,cacheDir = '/media/data_cifs/lu/Challenge/cache'):

        super(ICCVChallengeImporter,self).__init__(475.065948,-475.06585,315.944855,245.287079)

        self.path = path
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.numJoints =21
        self.scales = {'training': 1., 'test_1': 1., 'test_2': 0.83, 'test': 1., 'train_synth': 1.,
                       'test_synth_1': 1., 'test_synth_2': 0.83, 'test_synth': 1.}
        '''
        Will not be needed for the challenge dataset
        '''
        self.restrictedJoints = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]

    def loadDepthMap(self,filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.imread(filename)
        imgdata = np.asarray(img*65535,np.float32)
        '''        
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r, np.int32)
        g = np.asarray(g, np.int32)
        b = np.asarray(b, np.int32)
        dpt = np.bitwise_or(np.left_shift(g, 8), b)
        imgdata = np.asarray(dpt, np.float32)
        '''
        return imgdata

    def augmentCrop(self, img, gt3Dcrop, com, hd, cube, M, sigma_com=5., sigma_sc=0.05):
        # possible augmentation modes
        aug_modes = ['rot','scale','trans']
        # inplane rotation angle
        rot = np.random.uniform(0,360)
        # scaling factor
        sc = abs(1. + np.random.randn() * sigma_sc)

        # translation offset
        off = np.random.randn(3)*sigma_com
        # pick an augmentation method
        mode = np.random.randint(0,len(aug_modes))

        if aug_modes[mode] == 'rot':
            imgD, new_joints3D, _ = hd.rotateHand(img.astype('float32'), cube, com, rot, gt3Dcrop, pad_value=0)
        elif aug_modes[mode] == 'scale':
            imgD, new_joints3D, cube, M = hd.scaleHand(img.astype('float32'),cube,com,sc,gt3Dcrop,M)
        elif aug_modes[mode] == 'trans':
            imgD, new_joints3D, com, M = hd.transHand(img.astype('float32'), cube,com,off,gt3Dcrop,M)
        else:
            print('Such an augmentation method has not be implemented')
            exit(0)

        return imgD, new_joints3D, com, np.asarray(cube), M

    def jntsXYZtoUVD(self, jnts_xyz):
        jnts_uvd = np.zeros((jnts_xyz.shape[0], 3), np.float32)
        for j in range(jnts_xyz.shape[0]):
            jnts_uvd[j] = self.joint3DToImg(jnts_xyz[j])
        return jnts_uvd

    def loadGroundtruths(self,path_to_file):
        '''
        Load the joints location data specified in the annotation file
        This is a particular format for the ICCV Hands Challenge
        :param path_to_file: location of the annotation file
        :return: jnts (joint locations)
        '''
        jnts = []
        names = []
        with open(path_to_file) as f:
            for line in f:
                l = line.split('\t')
                names.append(l[0])
                jj = 1
                joint = np.zeros([self.numJoints, 3], np.float32)

                for j in range(0, self.numJoints):
                    joint[j, 0] = float(l[jj])
                    jj += 1
                    joint[j, 1] = float(l[jj])
                    jj += 1
                    joint[j, 2] = float(l[jj])
                    jj += 1
                jnts.append(joint)
        return np.asarray(names),np.asarray(jnts,dtype=np.float32)

    def loadAugSequence(self,seqName,  Nmax = float('inf'),shuffle = False, rng = None, docom = False,allJoints=False, num_aug=16):

        config = {'cube':(300,300,300)}
        config['cube'] = [s*self.scales[seqName] for s in config['cube']]

        if Nmax is float('inf'):
            pickleCache = '{}/{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName,allJoints)
        else:
            pickleCache = '{}/{}_{}_{}_cache_{}.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, allJoints,Nmax)
        print(pickleCache)
        if self.useCache:
            if os.path.isfile(pickleCache):
                print("Loading cache data from {}".format(pickleCache))
                f = open(pickleCache,'rb')
                (seqName,data,config) = cPickle.load(f)
                f.close()

                #shuffle data
                if shuffle and rng is not None:
                    print("shuffling")
                    rng.shuffle(data)
                if not(np.isinf(Nmax)):
                    return NamedImgSequence(seqName,data[0:Nmax],config)
                else:
                    return NamedImgSequence(seqName,data,config)

        '''
        LOAD THE DATA SET!
        objdir -> data location
        trainlabels -> text file containing groundtruth labels (image names, XYZ)
        '''
        objdir = '{}{}/images'.format(self.path,seqName)
        trainlabels = '{}/Training_Annotation.txt'.format(self.path)

        image_names, jnts_xyz = self.loadGroundtruths(trainlabels)
        '''
        Get the pertinent joints
        '''
        if allJoints:
            eval_idxs = np.arange(self.numJoints)
        else:
            eval_idxs = self.restrictedJoints

        '''
        Display a status bar to denote progress                                                                                                                                                                                                                                                                                  
        '''
        txt= 'Loading {}'.format(seqName)
        pbar = pb.ProgressBar(maxval=jnts_xyz.shape[0],widgets=[txt,pb.Percentage(),pb.Bar()])
        pbar.start()

        data = []

        '''
        Loop through every image
        '''
        image_index=0
        for im in image_names:
            # depth image name
            dptFileName='{}/{}'.format(objdir,im)
            print(dptFileName)
            # if image does not exist
            if not os.path.isfile(dptFileName):
                print("File {} does not exist!").format(dptFileName)
                image_index += 1
                continue
            dpt = self.loadDepthMap(dptFileName)

            # normalized joints in 3d coordinates for this image (XYZ)
            gt3Dorig = np.asarray(jnts_xyz[image_index,eval_idxs,:],dtype=np.float32)
            # joints in image coordinates (UVD)
            gtorig = self.joints3DToImg(gt3Dorig)
            #showAnnotatedDepth(ICVLFrame(dpt, gtorig, gtorig, 0, gt3Dorig, gt3Dorig, 0, dptFileName, ''))

            '''
            The augmentation process starts here!
            '''
            # Detect hand
            hd = HandDetector(dpt, self.fx, self.fy, importer=self)
            if not hd.checkImage(1):
                print("Skipping image {}, no content".format(dptFileName))
                image_index += 1
                continue

            '''
            if allJoints:
                hand_com = gtorig[0]
            else:
                hand_com = gtorig[0]
            '''
            hand_com = np.sum(gtorig,axis=0)/self.numJoints
            try:
                dpt, M, com = hd.cropArea3D(hand_com, size=config['cube'])
            except UserWarning:
                print ("Skipping image {}, no hand detected".format(dptFileName))
                continue

            com3D = self.jointImgTo3D(com)
            '''
            Normalize to com
            '''
            gt3Dcrop = gt3Dorig - com3D  # normalize to com
            gtcrop = np.zeros((gtorig.shape[0], 3), np.float32)
            for joint in range(gtorig.shape[0]):
                t = transformPoint2D(gtorig[joint], M)
                gtcrop[joint, 0] = t[0]
                gtcrop[joint, 1] = t[1]
                gtcrop[joint, 2] = gtorig[joint, 2]

            #showAnnotatedDepth(ICVLFrame(dpt, gtorig, gtcrop, M, gt3Dorig, gt3Dcrop, com3D, dptFileName, ''))
            '''
            Performing the augmentations
            '''
            for aug in range(num_aug):
                img, gt3DAug, com3dAug, cubeAug, M_aug = self.augmentCrop(dpt, gt3Dorig, com, hd, config['cube'],M)
                jnts = trans3DsToImg(gt3DAug-com3D, com3D, M) #gt3DAug
                #showAnnotatedDepth(ICVLFrame(img, gtorig, jnts, M, gt3Dorig, gt3DAug, com3dAug, dptFileName, ''))
                data.append(ICVLFrame(img.astype(np.float32),gtorig,jnts,M,gt3Dorig,gt3DAug-com3D,com3D,dptFileName,''))

            pbar.update(image_index)
            image_index+=1

            #early stop
            if len(data)>=Nmax:
                break

        pbar.finish()
        print("loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((seqName, data, config), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        # shuffle data
        if shuffle and rng is not None:
            print("shuffling")
            rng.shuffle(data)

        return NamedImgSequence(seqName,data,config)