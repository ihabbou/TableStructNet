import os
import os.path as osp
import mmcv
from glob import glob
from annotation_loader import parse_tables_from_xml

img_exts = [".bmp", ".jpg", ".jpeg", ".png", ".tiff"]

#https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html?highlight=coco#coco-annotation-format
#or https://github.com/open-mmlab/mmdetection/tree/master/tools/convert_datasets
def convert_icdar2019_to_coco(ann_files_path, out_file, img_prefix):

    cat2label = {k: i for i, k in enumerate(['table', 'cell'])}

    annotations = []
    images = []
    obj_count = 0
    
    image_list = [osp.basename(fn) 
                    for fn in glob(osp.join(img_prefix, '*.*')) 
                    if osp.splitext(fn.lower())[1] in img_exts]
    
    for idx, image_fn in enumerate(mmcv.track_iter_progress(image_list)):
        image_id = osp.splitext(osp.basename(image_fn))[0] 
        #filename = f'{image_id}.jpg' # TODO check this vs img_path
        filename = image_fn
        img_path = osp.join(img_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        # load annotations
        xml_path = f'{ann_files_path}/{image_id}.xml'
        tables = parse_tables_from_xml(xml_path)

        for table in tables:
            bbox = table.bbox
            area = (bbox[2]) * (bbox[3])
            poly = table.bounds
            
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=cat2label['table'],
                bbox=bbox,
                area=area,
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':cat2label[label], 'name': label} 
            for label in cat2label])
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    mmcv.dump(coco_format_json, out_file)

def restructure_ICDAR2019_dataset(root, out_dir, track="TRACKA", year="2014"):
    """
    
    """
    # start with training
    train_ann_files_path = osp.join(root, "training", track, "ground_truth")
    print(train_ann_files_path)
    train_img_prefix = osp.join(root, "training", track, "ground_truth")
    print(train_img_prefix)
    train_out_annotations = osp.join(out_dir, "annotations", f"instances_train{year}.json")
    print(train_out_annotations)

    print("Converting train annotations...")
    convert_icdar2019_to_coco(ann_files_path=train_ann_files_path, 
        out_file=train_out_annotations, 
        img_prefix=train_img_prefix)

    print("Moving train images...")
    train_img_dest = osp.join(out_dir, f"train{year}")
    os.makedirs(train_img_dest, exist_ok=True)
    train_img_list = [fn for fn in glob(osp.join(train_img_prefix, '*.*')) if osp.splitext(fn.lower())[1] in img_exts]
    for fn in mmcv.track_iter_progress(train_img_list):
        os.rename(fn, osp.join(train_img_dest, osp.basename(fn)))

    # val
    val_ann_files_path = osp.join(root, "test_ground_truth", track)
    print(val_ann_files_path)
    val_img_prefix = osp.join(root, "test", track)
    print(val_img_prefix)
    val_out_annotations = osp.join(out_dir, "annotations", f"instances_val{year}.json")
    print(val_out_annotations)
    
    print("Converting val annotations...")
    convert_icdar2019_to_coco(ann_files_path=val_ann_files_path, 
        out_file=val_out_annotations, 
        img_prefix=val_img_prefix)

    print("Moving val images...")
    val_img_dest = osp.join(out_dir, f"val{year}")
    os.makedirs(val_img_dest, exist_ok=True)
    val_img_list = [fn for fn in glob(osp.join(val_img_prefix, '*.*')) if osp.splitext(fn.lower())[1] in img_exts]
    for fn in mmcv.track_iter_progress(val_img_list):
        os.rename(fn, osp.join(val_img_dest, osp.basename(fn)))
    
    os.makedirs(osp.join(out_dir, "logs"), exist_ok=True)
