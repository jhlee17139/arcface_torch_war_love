from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.scrfd_weight = "../love_war_dataset/scrfd_weight/scrfd_34g.onnx"
config.inference_input = "../love_war_inference/input"
config.inference_output = "../love_war_inference/output"

config.margin_list = (1.0, 0.5, 0.0)
config.network = "r100"
config.resume = False
config.backbone_weight = "../love_war_dataset/weight/model.pt"
config.train_weight = "../love_war_dataset/classification_weight/18_epoch_model.pt"
config.output = "../love_war_dataset/classification_weight"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.94
config.weight_decay = 0.01
config.batch_size = 32
config.test_batch_size = 4
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "../love_war_dataset/train_face_img_folder"
config.classification_rec = "../love_war_dataset/train_face_img_folder"
config.test_rec = "../love_war_dataset/test_face_img_folder"
config.num_classes = 34
config.num_image = 1457
config.num_epoch = 30
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.drop_p = 0.5

config.labels = ['background', 'baeksomi', 'choijeongwon', 'choisehwan', 'choiyoungwan', 'hangeurim', 'hongseongsook', 'hongyeojin', 'jangeunbi', 'jeongnaon', 'jojeongrae', 'kangjiwoo', 'kangmoonyoung', 'kimdeokhyun', 'kimilran', 'kimjeongkyun', 'kimjimin', 'kimseonhyeok', 'kimsunyoung', 'kwakhyeonhwa', 'leejaewook', 'leejeongsoo', 'leejihoo', 'leejunwoo', 'leeseokwoo', 'leesol', 'leeyongyi', 'minjiyoung', 'moonbobae', 'parkjoohee', 'parkseonwoo', 'seokwonsoon', 'shinsomin', 'unkiho', 'yooncheolhyung']
config.labels_dist = [78, 35, 51, 18, 77, 48, 6, 28, 106, 61, 18, 62, 22, 59, 13, 16, 65, 12, 67, 21, 78, 19, 67, 42, 28, 4, 153, 11, 13, 60, 17, 31, 16, 35, 20]

'''
config.visualize_input = "../love_war_dataset/face_img_folder/choijeongwon/103000020014000100590011_0.jpg"
config.visualize_output = "../love_war_dataset/visualize/103000020014000100590011_0.jpg"
'''
'''
config.visualize_input = "../love_war_dataset/face_img_folder/choisehwan/103300020022000100540013_0.jpg"
config.visualize_output = "../love_war_dataset/visualize/103300020022000100540013_0.jpg"
'''
'''
config.visualize_input = "../love_war_dataset/face_img_folder/hongyeojin/103300020022000100650021_0.jpg"
config.visualize_output = "../love_war_dataset/visualize/103300020022000100650021_0.jpg"
'''
'''
config.visualize_input = "../love_war_dataset/face_img_folder/leejeongsoo/103100020021000100190007_0.jpg"
config.visualize_output = "../love_war_dataset/visualize/103100020021000100190007_0.jpg"
'''
'''
config.visualize_input = "../love_war_dataset/face_img_folder/leejeongsoo/103100020021000100590017_0.jpg"
config.visualize_output = "../love_war_dataset/visualize/103100020021000100590017_0.jpg"
'''

config.visualize_input = "../love_war_dataset/face_img_folder/leejeonghoon/103000020014000100940013_0.jpg"
config.visualize_output = "../love_war_dataset/visualize/103000020014000100940013_0.jpg"



