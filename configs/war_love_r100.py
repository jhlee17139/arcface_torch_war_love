from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r100"
config.resume = False
config.output = "../love_war_dataset/weight"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-5
config.batch_size = 16
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "../love_war_dataset/train_face_img_folder_no_bg"
config.num_classes = 34
config.num_image = 1379
config.num_epoch = 9
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
