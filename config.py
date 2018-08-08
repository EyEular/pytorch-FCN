import warnings

class DefaultConfig(object):
    model = 'FCN'

# config of data before training
    train_data_root = '/home/eulring/Dataset/kaggleSalt'
    train_label_list = '/home/eulring/Dataset/kaggleSalt/train.csv'

    
# config of data in training
    batch_size = 2
    use_gpu = True
    num_workers = 4
    print_freq = 10
    
    debug_mode = False
    #debug_mode = True

# config of training hyperparameter
    max_epoch = 5
    lr = 0.0001
    lr_decay = 0.5
    momentum = 0
    weight_decay = 0e-5






def parse(self, kwargs):
# update the config according to kwargs
    for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
