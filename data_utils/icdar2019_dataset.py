import copy
import os.path as osp
from glob import glob

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

from annotation_loader import parse_tables_from_xml


@DATASETS.register_module()
class ICDAR2019Dataset(CustomDataset):

    CLASSES = ('table', 'cell')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        # TODO fix extention
        image_list = [osp.splitext(osp.basename(fn))[0] 
                        for fn in glob(osp.join(self.ann_file, '*.jpg'))]

        
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
