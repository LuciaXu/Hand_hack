from os.path import join as pjoin

class handConfig(object):
    def __init__(self):
        #dir setting
        self.results_dir = '/media/data_cifs/lakshmi/results/'
        self.model_output = '/media/data_cifs/lakshmi/models/'
        self.model_input = ''
        self.train_summaries = '/media/data_cifs/lakshmi/summaries/12sept'
        self.tfrecord_dir = '/media/data_cifs/lakshmi/tfrecords/'
        self.train_tfrecords = 'train.tfrecords'
        self.val_tfrecords = 'val.tfrecords'
        self.test_tfrecords = 'test.tfrecords'
        self.vgg16_weight_path = pjoin(
            '/media/data_cifs/clicktionary/',
            'pretrained_weights',
            'vgg16.npy')

        #model setting
        self.model_type = 'vgg_regression_model_4fc'
        self.epochs = 300
        self.image_target_size = [128,128,1]
        self.label_shape = 36
        self.train_batch = 1
        self.val_batch= 1
        self.initialize_layers = ['fc6', 'fc7', 'pre_fc8', 'fc8']
        self.fine_tune_layers = ['fc6', 'fc7', 'pre_fc8', 'fc8']
        self.batch_norm = ['conv1','fc1','fc2']
        self.wd_layers = ['fc6', 'fc7', 'pre_fc8']
        self.wd_penalty = 0.005
        self.optimizier = 'adam'
        self.lr = 1e-4  # Tune this -- also try SGD instead of ADAm
        self.hold_lr = self.lr / 2
        self.keep_checkpoints = 100

        #training setting
        self.num_classes = 108  # there are 36 * 3 (x/y/z) joint coors
        #self.num_classes = 42
