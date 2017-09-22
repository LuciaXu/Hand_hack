import tensorflow as tf
import numpy as np
import os
import cv2
from utils import *

def deg2rad(deg):
    return deg * (np.pi/180.0)

def read_and_decode(filename_queue,target_size,label_shape,data_augment):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features = {
        'label':tf.FixedLenFeature([],tf.string),
        'image':tf.FixedLenFeature([],tf.string),
        'com3D':tf.FixedLenFeature([],tf.string),
        'M':tf.FixedLenFeature([],tf.string)
    })
    print('in read and decode')
    #convert from a scalor string tensor
    label = tf.decode_raw(features['label'],tf.float32)

    image = tf.decode_raw(features['image'],tf.float32)

    com3D = tf.decode_raw(features['com3D'],tf.float32)

    M = tf.decode_raw(features['M'],tf.float32)

    #Need to reconstruct channels first then transpose channels
    image = tf.reshape(image,np.asarray(target_size))
    label.set_shape(label_shape)
    com3D = tf.reshape(com3D,(3,))
    M=tf.reshape(M,(3,3))

    if (data_augment):
        '''
        Augmentations happen here! Take each sample and pick an aug mode 
        and modify the image and the respective label
        '''
        dim = image.get_shape().as_list()
        label_dims = label.get_shape().as_list()
        label = tf.reshape(label, [label_dims[0] / 3, 3])

        # call the wrapper
        label, image, com3D, M = tf.py_func(augment_sample,[label,image,com3D,M,dim],[tf.float32,tf.float32,tf.float32,tf.float32])
        # reshape outputs to compatible states
        label = tf.reshape(label,[label_shape,])
        image = tf.reshape(image, np.asarray(target_size))

        com3D = tf.reshape(com3D, (3,))
        M = tf.reshape(M, (3, 3))
        '''
        Ends here!
        '''

    #print("com3D shape:{}".format(com3D.get_shape().as_list()))
    #print("M shape:{}".format(M.get_shape().as_list()))

    #print(" after reshape image shape:{}".format(image.get_shape().as_list()))
    #print(" after reshape lable shape:{}".format(label.get_shape().as_list()))

    #image_rgb = tf.stack([image,image,image],axis=2)

    return label, image,com3D,M
    #return label,image


def inputs(tfrecord_file,num_epochs,image_target_size,label_shape,batch_size,data_augment=False):
    print('in input!')
    with tf.name_scope('input'):
        if os.path.exists(tfrecord_file) is False:
            print("{} not exists".format(tfrecord_file))
        # returns a queue. adds a queue runner for the queue to the current graph's QUEUE_RUNNER
        filename_queue = tf.train.string_input_producer([tfrecord_file],num_epochs=num_epochs)
        label, image, com3D, M = read_and_decode(filename_queue = filename_queue,target_size=image_target_size,label_shape = label_shape,data_augment = data_augment)
        # return a list or dictionary. adds 1) a shuffling queue into which tensors are enqueued; 2) a dequeue_many operation to create batches
        # from the queue 3) a queue runner to QUEUE_RUNNER collection , to enqueue the tensors.
        data,labels,com3Ds,Ms = tf.train.shuffle_batch([image,label,com3D,M],batch_size=batch_size,num_threads=2,capacity=100+3*batch_size,min_after_dequeue=1)
        # print(" in input image shape:{}".format(data.get_shape))
        # print(" in input lable shape:{}".format(labels.get_shape))
        #print(" in input com3Ds shape:{}".format(com3Ds.get_shape))
        #print(" in input Ms shape:{}".format(Ms.get_shape))
    return data,labels,com3Ds,Ms
    #return image, label,com3D,M

