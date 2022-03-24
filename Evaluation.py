import json
import pickle5 as pickle
from apmeter import APMeter
import numpy as np
from utils import *


def make_gt(gt_file, logits, num_classes=157):
    gt_new = {}
    vid_length = {}
    fps_seg = {}
    with open(gt_file, 'r') as f:
        gt = json.load(f)

    i = 0
    gt_len=0
    for vid in gt.keys():
        if gt[vid]['subset'] != "testing":
            continue
        else:
            gt_len=gt_len+1
    for vid in gt.keys():
        if gt[vid]['subset'] != "testing":
            continue
        if vid not in logits.keys():
            continue
        num_pred = logits[vid].shape[1]

        label = np.zeros((num_pred, num_classes), np.float32)

        fps = float(num_pred / float(gt[vid]['duration']))
        for ann in gt[vid]['actions']:
            for fr in range(0, num_pred, 1):
                if fr / fps > ann[1] and fr / fps < ann[2]:
                    label[fr, ann[0]] = 1
        gt_new[vid]=label
        vid_length[vid]=gt[vid]['duration']
        fps_seg[vid]=fps
        i += 1
    return gt_new,vid_length,fps_seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-pkl_path', type=str)  # './test.pkl'
    args = parser.parse_args()

    pkl_path = args.pkl_path

    gt_file = './data/charades.json'
    classes = 157

    pkl = open(pkl_path, 'rb')

    logits = pickle.load(pkl)

    gt_new,vid_len,fps_seg=make_gt(gt_file,logits,classes)

    # Compute mAP
    apm = APMeter()
    sampled_apm=APMeter()
    first_idx=0
    idx=0
    pred_probs=[]
    gt_labels=[]

    for vid in gt_new.keys():
        idx=idx+1
        logit = np.transpose(logits[vid], (1, 0))

        apm.add(logit, gt_new[vid])
        sampled_25_inference(logit,gt_new[vid],sampled_apm)
        pred_probs.append(logit)
        gt_labels.append(gt_new[vid])

    # per-frame mAP
    val_map = 100 * apm.value().mean()
    sample_val_map = 100 *sampled_apm.value().mean()
    print ("Test Frame-based map", val_map)
    print ("25 sampled Frame-based map", sample_val_map)
    print ("APs for the classes",100 * apm.value())

    # action-conditional metrics for different t
    # t=0
    prec0, re0, ns0, map0 = conditional_metric(pred_probs, gt_labels, t=0, avg=True)
    fs0 = get_f1(prec0, re0)  # action conditional f1-score
    print('Precision(c_i|c_j,0)=', prec0)
    print('Recall(c_i|c_j,0)=', re0)
    print('F1Score(c_i|c_j,0)=', fs0)
    print('mAP(c_i|c_j,0)=', map0)

    # t=20
    prec20, re20, ns20, map20 = conditional_metric(pred_probs, gt_labels, t=20, avg=True)
    fs20 = get_f1(prec20, re20)  # action conditional f1-score
    print('Precision(c_i|c_j,20)=', prec20)
    print('Recall(c_i|c_j,20)=', re20)
    print('F1Score(c_i|c_j,20)=', fs20)
    print('mAP(c_i|c_j,20)=', map20)

    # t=40
    prec40, re40, ns40, map40 = conditional_metric(pred_probs, gt_labels, t=40, avg=True)
    fs40 = get_f1(prec40, re40)  # action conditional f1-score
    print('Precision(c_i|c_j,40)=', prec40)
    print('Recall(c_i|c_j,40)=', re40)
    print('F1Score(c_i|c_j,40)=', fs40)
    print('mAP(c_i|c_j,40)=', map40)


