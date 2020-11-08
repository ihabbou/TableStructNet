import copy
import os.path as osp
from glob import glob

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

from annotation_loader import parse_tables_from_xml
"""

# Modify dataset type and path
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

"""
@DATASETS.register_module()
class ICDAR2019Dataset(CustomDataset):

    CLASSES = ('table', 'cell')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = [osp.splitext(osp.basename(fn))[0] 
                        for fn in glob(self.ann_file+'*.jpg')]

        
        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            # TODO check extentions
            filename = f'{self.img_prefix}/{image_id}.jpg'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{image_id}.jpg', width=width, height=height)
    
            # load annotations
            xml_path = f'{self.ann_file}/{image_id}.xml'
            tables = parse_tables_from_xml(xml_path)
            # TODO do we want cell bbxs for table detector?
            bboxes = [table.bbox for table in tables]
            bbox_names = ['table' for table in tables] # TODO remove?
            
            gt_bboxes = []
            gt_labels = []
            # gt_bboxes_ignore = []
            # gt_labels_ignore = []
    
            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                # else:
                #     gt_labels_ignore.append(-1)
                #     gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long)#,
                # bboxes_ignore=np.array(gt_bboxes_ignore,
                #                        dtype=np.float32).reshape(-1, 4),
                # labels_ignore=np.array(gt_labels_ignore, dtype=np.long)
            )

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos
