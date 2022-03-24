import torch
import numpy as np
import argparse
import os

from easydict import EasyDict as edict
from models import Yolov4
from cfg import Cfg
from PIL import Image
from tool.utils import do_detect, bbox_iou
from pprint import pprint


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(
        description='Evaluate the model trained to detec trees',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-f', '--load', dest='load', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1', help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data_dir', type=str, default=None, help='dataset dir', dest='dataset_dir')
    parser.add_argument('-n', '--classes', type=int,default=1,help='dataset classes number')
    parser.add_argument('-c', '--model_config', type=str, default='cfg/yolov4.cfg',
                            help='model config file to load', dest='model_config')
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)


def evaluate_on_trees(pred_boxes, gt_boxes):
    preds_count = 0
    pred_num = 0
    for _, pred in pred_boxes.items():
        preds_count += len(pred)
    pred_assignments = dict.fromkeys(range(preds_count))

    for key, pred in pred_boxes.items():
        for pred_box in pred:
            ious = [bbox_iou(pred_box, gt_box) for gt_box in gt_boxes[key]]
            highest_iou_key = ious.index(max(ious))
            if ious[highest_iou_key] > 0.5:
                if not any([
                    True for _, assignment in pred_assignments.items() 
                    if assignment is not None and gt_boxes[key][highest_iou_key] == assignment[1]
                ]):
                    pred_assignments[pred_num] = (key, gt_boxes[key][highest_iou_key])
                    pred_num += 1
            else:
                pred_num += 1

    tp, fp, fn, gt_count = 0, 0, 0, 0
    for _, pred in gt_boxes.items():
        gt_count += len(pred)
    for _, assignment in pred_assignments.items():
        if assignment is not None:
            tp += 1
        else:
            fp +=1

    fn = gt_count - tp

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp) 

    recall = tp / (tp + fn) 
    f1 = 2*((recall*precision)/(recall+precision + 0.00000001))
    return precision, recall, f1

 
def parse_annotations(test_dataset_path):
    annotations_path = test_dataset_path + "/_annotations.txt"
    annotations = open(annotations_path, "r")
    content = annotations.read()
    gt_preds = content.split("\n")
    gt_annotations = {}
    for preds in gt_preds:
        pred = preds.split(" ")
        gt_annotations[pred[0]] = []
        for index, gt_boxes in enumerate(pred):
            if index!=0:
                boxes = [int(s) for s in gt_boxes.split(",")]
                gt_annotations[pred[0]].append(boxes)
    return gt_annotations


def test(model, test_dataset_path):
    if torch.cuda.is_available():
        use_cuda = 1
    else:
        use_cuda = 0
    pred_boxes = {}
    for filename in os.listdir(test_dataset_path):
        if filename.endswith(".jpg"):
            img_path = test_dataset_path + "/" + filename
            key = Image.open(img_path).convert('RGB')
            sized = key.resize((608, 608))
            boxes = do_detect(model, sized, 0.05, 1, 0.4, use_cuda)
            for box in boxes:
                x1 = int((box[0] - box[2] / 2.0) * key.width)
                y1 = int((box[1] - box[3] / 2.0) * key.height)
                x2 = int((box[0] + box[2] / 2.0) * key.width)
                y2 = int((box[1] + box[3] / 2.0) * key.height)
                box[0] = x1
                box[1] = y1
                box[2] = x2
                box[3] = y2
            pred_boxes[filename] = boxes

    gt_boxes = parse_annotations(test_dataset_path)
    precisions = []
    recalls = []
    f1s = []
    thresholds = np.arange(start=0.05, stop=1.00, step=0.05)
    for threshold in thresholds:
        filtered_pred_boxes = {}
        for key, boxes in pred_boxes.items():
            box_list = []
            for box in boxes:
                if box[4] > threshold:
                    box_list.append(box)
            filtered_pred_boxes[key] = box_list
        precision, recall, f1 = evaluate_on_trees(filtered_pred_boxes, gt_boxes)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precisions.append(1)
    recalls.append(0)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    average_precision = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    metrics = {
        "thresholds": thresholds,
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "AP": average_precision
    }
    return metrics
    

def main():
    cfg = get_args(**Cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Yolov4(n_classes=cfg.classes)

    pretrained_dict = torch.load(cfg.load, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    model.eval()
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    metrics = test(model, cfg.dataset_dir)
    pprint(metrics)


if __name__ == "__main__":
    main()