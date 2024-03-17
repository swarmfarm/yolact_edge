pip install --quiet git+https://github.com/haotian-liu/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"

current_datetime=$(date +"%Y%m%d_%H%M%S")
results_dir=/notebooks/yolact_edge/results/swarmfarm

mkdir -p $results_dir
mkdir -p ${results_dir}/pretrained_coco_conf_30
mkdir -p ${results_dir}/pretrained_coco_conf_30/images
mkdir -p ${results_dir}/pretrained_coco_conf_30/info

python eval.py --disable_tensorrt \
    --output_coco_json \
    --trained_model=weights/yolact_edge_54_800000.pth \
    --score_threshold=0.3 \
    --top_k=100 \
    --images=./data/swarmfarm/annotation:${results_dir}/pretrained_coco_conf_30/images \
    --bbox_det_file=${results_dir}/pretrained_coco_conf_30/info/bbox_detections.json \
    --mask_det_file=${results_dir}/pretrained_coco_conf_30/info/mask_detections.json