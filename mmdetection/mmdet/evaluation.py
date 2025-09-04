from mmdet.evaluation.functional import eval_map
from pycocotools.coco import COCO
import json

# 加载检测结果
with open('/home/yuanyang/data_extract/mmdetection/result.bbox.json', 'r') as f:
    det_results = json.load(f)

# 加载COCO格式标注
coco = COCO('/home/yuanyang/data_extract/data/pmc_2022/pmc_coco/plots_detection/bar_test.json')

# 转换为mmdetection需要的格式
annotations = []
for img_id in coco.getImgIds():
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_anns = []
    for ann in anns:
        img_anns.append({
            'bbox': ann['bbox'],
            'category_id': ann['category_id']
        })
    annotations.append(img_anns)

# 验证数据一致性
assert len(det_results) == len(annotations), \
    f"检测结果({len(det_results)})和标注({len(annotations)})数量不匹配"

# 计算mAP
thr = 0.5
mean_ap, eval_results = eval_map(
    det_results=det_results,
    annotations=annotations,
    iou_thr=thr
)

print(f"mAP@{thr}: {mean_ap}")