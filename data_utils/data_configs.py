

def add_icdar2019_dataset(cfg):
    
    cfg.dataset_type = 'ICDAR2019Dataset'
    cfg.data_root = 'ICDAR2019_cTDaR-master/'

    cfg.data.test.type = 'ICDAR2019Dataset'
    cfg.data.test.data_root = 'training/TRACKA/'
    cfg.data.test.ann_file = 'ground_truth'
    cfg.data.test.img_prefix = 'ground_truth'

    cfg.data.train.type = 'ICDAR2019Dataset'
    cfg.data.train.data_root = 'training/TRACKA/'
    cfg.data.train.ann_file = 'ground_truth'
    cfg.data.train.img_prefix = 'ground_truth'

    cfg.data.val.type = 'ICDAR2019Dataset'
    cfg.data.val.data_root = ''
    cfg.data.val.ann_file = 'test_ground_truth/TRACKA'
    cfg.data.val.img_prefix = 'test/TRACKA'
    
    return cfg