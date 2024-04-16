from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super(MyConfig, self).__init__()
        # Dataset
        self.dataset = 'cityscapes'
        self.data_root = '/content/cityscapes/Cityscape'
        self.num_class = 20
        
        # Model
        self.model = 'erfnet'
        
        # Training
        self.total_epoch = 30 #20
        self.train_bs = 8
        self.loss_type = 'ce' # ohem
        self.optimizer_type = 'adam'
        self.logger_name = 'seg_trainer'
        self.use_aux = False

        # Validating
        self.val_bs = 10
        
        # Testing
        self.is_testing = False
        self.test_bs = 8
        self.test_data_folder = '/content/test'
        self.load_ckpt_path = '/path/to/your/inference/checkpoint'
        self.save_mask = True
        
        # Training setting
        self.use_ema = False
        
        # Augmentation
        self.crop_size = 768
        self.randscale = [-0.5, 1.0]
        self.scale = 1.0
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.h_flip = 0.5
        
        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = '/path/to/your/teacher/checkpoint'
        self.teacher_model = 'smp'
        self.teacher_encoder = 'resnet101'
        self.teacher_decoder = 'deeplabv3p'
