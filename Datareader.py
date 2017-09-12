from Importer import NYUImporter
from dataset import NYUDataset
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from check_fun import showdepth,trans3DsToImg,showImagefromArray,showImageJoints

def bytes_feature(values):
    """Encodes an float matrix into a byte list for a tfrecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def encode_example(im,label,com3D,M):
    feature = {
        'label':bytes_feature(label.tostring()),
        'image':bytes_feature(im.tostring()),
        'com3D':bytes_feature(com3D.tostring()),
        'M':bytes_feature(M.tostring())
    }
    example = tf.train.Example(features = tf.train.Features(feature = feature))
    return example.SerializeToString()

def create_tf_record(depth_files,lable_files,tf_file,config,com3D_files,M_files):
    print(os.path.abspath(tf_file))

    with tf.python_io.TFRecordWriter(tf_file) as tf_writer:
        for i, (depth, label,com3D,M) in tqdm(enumerate(zip(depth_files,lable_files,com3D_files,M_files)),total=len(depth_files)):
            example = encode_example(depth,label,com3D,M)
            tf_writer.write(example)


def Datareader(config):
    print("create training data")
    cwd = os.getcwd()
    print(cwd)
    rng = np.random.RandomState(23455)
    ds = NYUImporter('/media/data_cifs/lu/NYU/dataset')
    Seq1= ds.loadAugSequence('train', shuffle=True, rng=rng, docom=True,allJoints=True,Nmax=10)
    trainSeqs = [Seq1]

    trainDataSet = NYUDataset(imgSeqs = trainSeqs)
    #trainDataSet.check()
    train_data, train_gt3D, seqconfig, train_com3D,train_M = trainDataSet.imgStackDepthOnly('train')


    # print("shape of train_data {}".format(train_data.shape))
    # print("shape of train lable {}".format(train_gt3D.shape))
    # print("seqconfig:{}".format(seqconfig))

    # #check if the data is already now
    # for check_num in range(val_data.shape[0]):
    #     if (check_num > 10) and (check_num<15):
    #         check_data = val_data[check_num]
    #         check_gt3D = val_gt3D[check_num]
    #         check_com3D = val_com3D[check_num]
    #         check_M = val_M[check_num]
    #         print("shape of check data:{}".format(check_data.shape))
    #         print("shape of check gt3D:{}".format(check_gt3D.shape))
    #         print("shape of check com3D:{}".format(check_com3D.shape))
    #         print("shape of check M:{}".format(check_M.shape))
    #         jtorig = check_gt3D * seqconfig['cube'][2] / 2.
    #         jcrop = trans3DsToImg(jtorig, check_com3D, check_M)
    #         im_g = check_data.reshape([128,128])
    #         # showImageLable(im_g,jcrop)
    #         showImageJoints(im_g, jcrop)

    print('Get {} training data.'.format(train_data.shape[0]))
    create_tf_record(train_data,train_gt3D,os.path.join(config.tfrecord_dir,config.train_tfrecords),config,train_com3D,train_M)
    print('create testing data and validation data')
    Seq2 = ds.loadSequence('test', shuffle=True, rng=rng,docom=True,allJoints=True)
    testSeqs = [Seq2]
    testDataSet = NYUDataset(imgSeqs=testSeqs,val_prop=0.3)
    test_data, test_gt3D,val_data,val_gt3D,testconfig,test_com3D,val_com3D,test_M,val_M = testDataSet.imgStackDepthOnly('test')
    create_tf_record(val_data, val_gt3D, os.path.join(config.tfrecord_dir, config.val_tfrecords), config, val_com3D,
                     val_M)
    create_tf_record(test_data, test_gt3D, os.path.join(config.tfrecord_dir, config.test_tfrecords), config,test_com3D,test_M)
    print('Get {} test data'.format(test_data.shape[0]))
    print('Get {} validation data'.format(val_data.shape[0]))

    return seqconfig


