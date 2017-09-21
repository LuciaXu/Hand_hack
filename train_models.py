import os
import re
import time
from datetime import datetime
import tensorflow as tf
from data_loader import inputs
from check_fun import showdepth, showImagefromArray,showImageLable,trans3DsToImg,showImageLableCom,showImageJoints,showImageJointsandResults
#from tf_fun import regression_mse, correlation, make_dir, \
#    fine_tune_prepare_layers, ft_optimizer_list
from pose_evaluation import getMeanError,getMeanError_np,getMean_np,getMeanError_train
import numpy as np
import cPickle
#from checkpoint import  list_variables


def check_image_label(im, jts, com, M,cube_22,allJoints=False,line=False):
    relen=len(jts)/3
    jt = jts.reshape((relen, 3))
    jtorig = jt * cube_22
    jcrop = trans3DsToImg(jtorig, com, M)
    showImageJoints(im,jcrop,allJoints=allJoints,line=line)

def test_input_full(config,seqconfig):
    train_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    val_data = os.path.join(config.tfrecord_dir, config.val_tfrecords)
    with tf.device('/cpu:0'):
        train_images,train_labels,com3Ds,Ms = inputs(tfrecord_file = train_data,
                                           num_epochs=config.epochs,
                                           image_target_size = config.image_target_size,
                                           label_shape=config.num_classes,
                                           batch_size =config.train_batch,
                                                     data_augment=True)
        val_images, val_labels, val_com3Ds, val_Ms = inputs(tfrecord_file=val_data,
                                                            num_epochs=config.epochs,
                                                            image_target_size=config.image_target_size,
                                                            label_shape=config.num_classes,
                                                            batch_size=1)
        label_shaped = tf.reshape(train_labels,[config.train_batch,config.num_classes/3,3])
        error = getMeanError(label_shaped,label_shaped)
        val_label_shaped = tf.reshape(val_labels, [1, config.num_classes/3, 3])
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        step =0
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                image_np,image_label,image_coms,image_Ms,train_error = sess.run([train_images,train_labels,com3Ds,Ms,error])
                print step
                #print image_np.shape
                #print train_error

                val_image_np, val_image_label, val_image_coms, val_image_Ms= sess.run(
                    [val_images, val_labels, val_com3Ds, val_Ms])


                if (step > 0) and (step <2):

                    for b in range(config.train_batch):
                        im = image_np[b]
                        image_com = image_coms[b]
                        image_M = image_Ms[b]

                        jts = image_label[b]
                        print("shape of jts:{}".format(jts.shape))
                        im = im.reshape([128,128])
                        check_image_label(im,jts,image_com,image_M,seqconfig['cube'][2] / 2.,allJoints=True,line=True)

                    val_im = val_image_np[0]
                    print("val_im shape:{}".format(val_im.shape))
                    val_image_com = val_image_coms[0]
                    val_image_M = val_image_Ms[0]
                    # print("shape of im:{}".format(im.shape))
                    val_jts = val_image_label[0]
                    val_im = val_im.reshape([128, 128])
                    check_image_label(val_im, val_jts, val_image_com, val_image_M, seqconfig['cube'][2] / 2.,allJoints=True,line=True)

                step += 1

        except tf.errors.OutOfRangeError:
            print("Done. Epoch limit reached.")
        finally:
            coord.request_stop()
        coord.join(threads)

def test_input_tree(config,seqconfig):
    train_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    val_data = os.path.join(config.tfrecord_dir, config.val_tfrecords)
    with tf.device('/cpu:0'):
        train_images,train_labels,com3Ds,Ms = inputs(tfrecord_file = train_data,
                                           num_epochs=config.epochs,
                                           image_target_size = config.image_target_size,
                                           label_shape=config.num_classes,
                                           batch_size =config.train_batch)
        val_images, val_labels, val_com3Ds, val_Ms = inputs(tfrecord_file=val_data,
                                                            num_epochs=config.epochs,
                                                            image_target_size=config.image_target_size,
                                                            label_shape=config.num_classes,
                                                            batch_size=1)
        label_shaped = tf.reshape(train_labels,[config.train_batch,config.num_classes/3,3])
        split_lable=tf.split(label_shaped,36,axis=1)
        P_label_shaped = tf.concat(
            [split_lable[0], split_lable[1], split_lable[2], split_lable[3], split_lable[4], split_lable[5],
             split_lable[29], split_lable[30], split_lable[31], split_lable[32], split_lable[33], split_lable[34],
             split_lable[35]], axis=1)
        R_label_shaped = tf.concat(
            [split_lable[6], split_lable[7], split_lable[8], split_lable[9], split_lable[10], split_lable[11],
             split_lable[29], split_lable[30], split_lable[31], split_lable[32], split_lable[33], split_lable[34],
             split_lable[35]], axis=1)
        M_label_shaped = tf.concat(
            [split_lable[12], split_lable[13], split_lable[14], split_lable[15], split_lable[16], split_lable[17],
             split_lable[29], split_lable[30], split_lable[31], split_lable[32], split_lable[33], split_lable[34],
             split_lable[35]], axis=1)
        I_label_shaped = tf.concat(
            [split_lable[18], split_lable[19], split_lable[20], split_lable[21], split_lable[22], split_lable[23],
             split_lable[29], split_lable[30], split_lable[31], split_lable[32], split_lable[33], split_lable[34],
             split_lable[35]], axis=1)
        T_label_shaped = tf.concat(
            [split_lable[24], split_lable[25], split_lable[26], split_lable[27], split_lable[28],
             split_lable[29], split_lable[30], split_lable[31], split_lable[32], split_lable[33], split_lable[34],
             split_lable[35]], axis=1)
        P_label = tf.reshape(P_label_shaped, [config.train_batch, P_label_shaped.get_shape().as_list()[1] * 3])
        R_label = tf.reshape(R_label_shaped, [config.train_batch, R_label_shaped.get_shape().as_list()[1] * 3])
        M_label = tf.reshape(M_label_shaped, [config.train_batch, M_label_shaped.get_shape().as_list()[1] * 3])
        I_label = tf.reshape(I_label_shaped, [config.train_batch, I_label_shaped.get_shape().as_list()[1] * 3])
        T_label = tf.reshape(T_label_shaped, [config.train_batch, T_label_shaped.get_shape().as_list()[1] * 3])
        error = getMeanError(label_shaped,label_shaped)
        val_label_shaped = tf.reshape(val_labels, [1, config.num_classes/3, 3])
        val_split_lable = tf.split(val_label_shaped, 36, axis=1)
        val_P_label_shaped = tf.concat(
            [val_split_lable[0], val_split_lable[1], val_split_lable[2], val_split_lable[3], val_split_lable[4],
             val_split_lable[5],
             val_split_lable[29], val_split_lable[30], val_split_lable[31], val_split_lable[32], val_split_lable[33],
             val_split_lable[34],
             val_split_lable[35]], axis=1)
        val_R_label_shaped = tf.concat(
            [val_split_lable[6], val_split_lable[7], val_split_lable[8], val_split_lable[9], val_split_lable[10],
             val_split_lable[11],
             val_split_lable[29], val_split_lable[30], val_split_lable[31], val_split_lable[32], val_split_lable[33],
             val_split_lable[34],
             val_split_lable[35]], axis=1)
        val_M_label_shaped = tf.concat(
            [val_split_lable[12], val_split_lable[13], val_split_lable[14], val_split_lable[15], val_split_lable[16],
             val_split_lable[17],
             val_split_lable[29], val_split_lable[30], val_split_lable[31], val_split_lable[32], val_split_lable[33],
             val_split_lable[34],
             val_split_lable[35]], axis=1)
        val_I_label_shaped = tf.concat(
            [val_split_lable[18], val_split_lable[19], val_split_lable[20], val_split_lable[21], val_split_lable[22],
             val_split_lable[23],
             val_split_lable[29], val_split_lable[30], val_split_lable[31], val_split_lable[32], val_split_lable[33],
             val_split_lable[34],
             val_split_lable[35]], axis=1)
        val_T_label_shaped = tf.concat(
            [val_split_lable[24], val_split_lable[25], val_split_lable[26], val_split_lable[27], val_split_lable[28],
             val_split_lable[29], val_split_lable[30], val_split_lable[31], val_split_lable[32], val_split_lable[33],
             val_split_lable[34],
             val_split_lable[35]], axis=1)
        val_P_label = tf.reshape(val_P_label_shaped, [config.val_batch, val_P_label_shaped.get_shape().as_list()[1] * 3])
        val_R_label = tf.reshape(val_R_label_shaped, [config.val_batch, val_R_label_shaped.get_shape().as_list()[1] * 3])
        val_M_label = tf.reshape(val_M_label_shaped, [config.val_batch, val_M_label_shaped.get_shape().as_list()[1] * 3])
        val_I_label = tf.reshape(val_I_label_shaped, [config.val_batch, val_I_label_shaped.get_shape().as_list()[1] * 3])
        val_T_label = tf.reshape(val_T_label_shaped, [config.val_batch, val_T_label_shaped.get_shape().as_list()[1] * 3])
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        step =0
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                image_np,image_label,image_coms,image_Ms,train_error,P,R,M,I,T = sess.run([train_images,train_labels,com3Ds,Ms,error,P_label,R_label,\
                                                                                 M_label,I_label,T_label])
                print step
                print image_np.shape
                print train_error

                val_image_np, val_image_label, val_image_coms, val_image_Ms,val_P,val_R,val_M,val_I,val_T= sess.run(
                    [val_images, val_labels, val_com3Ds, val_Ms,val_P_label,val_R_label,val_M_label,val_I_label,val_T_label])

                #image = tf.split(image_np, 3, 3)[0]
                #print image.shape
                #print image_label.shape

                if (step > 0) and (step <2):

                    for b in range(config.train_batch):
                        im = image_np[b]
                        print("im shape:{}".format(im.shape))
                        image_com = image_coms[b]
                        image_M = image_Ms[b]
                        #print("shape of im:{}".format(im.shape))
                        jts = image_label[b]
                        im = im.reshape([128,128])
                        check_image_label(im,jts,image_com,image_M,seqconfig['cube'][2] / 2.,allJoints=True,line=True)
                        check_image_label(im,P[b],image_com,image_M,seqconfig['cube'][2] / 2.,line=False)
                        check_image_label(im, R[b], image_com, image_M, seqconfig['cube'][2] / 2., line=False)
                        check_image_label(im, M[b], image_com, image_M, seqconfig['cube'][2] / 2., line=False)
                        check_image_label(im, I[b], image_com, image_M, seqconfig['cube'][2] / 2., line=False)
                        check_image_label(im, T[b], image_com, image_M, seqconfig['cube'][2] / 2., line=False)

                    val_im = val_image_np[0]
                    print("val_im shape:{}".format(val_im.shape))
                    val_image_com = val_image_coms[0]
                    val_image_M = val_image_Ms[0]
                    # print("shape of im:{}".format(im.shape))
                    val_jts = val_image_label[0]
                    val_im = val_im.reshape([128, 128])
                    check_image_label(val_im, val_jts, val_image_com, val_image_M, seqconfig['cube'][2] / 2.,allJoints=True,line=True)
                    check_image_label(val_im, val_P[0], val_image_com, val_image_M, seqconfig['cube'][2] / 2.,line=False)
                    check_image_label(val_im, val_R[0], val_image_com, val_image_M, seqconfig['cube'][2] / 2.,
                                      line=False)
                    check_image_label(val_im, val_M[0], val_image_com, val_image_M, seqconfig['cube'][2] / 2.,
                                      line=False)
                    check_image_label(val_im, val_I[0], val_image_com, val_image_M, seqconfig['cube'][2] / 2.,
                                      line=False)
                    check_image_label(val_im, val_T[0], val_image_com, val_image_M, seqconfig['cube'][2] / 2.,
                                      line=False)
                step += 1

        except tf.errors.OutOfRangeError:
            print("Done. Epoch limit reached.")
        finally:
            coord.request_stop()
        coord.join(threads)