import torch._utils
import torch
import torch.nn.functional as F
import argparse
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import numpy as np


def softmax(x,ax):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=ax)


def conditional_t(y_pred, y_gt, y_gt_mask, thresh=0.5, avg=True):
    n_samples, n_classes = y_pred.shape
    assert y_pred.shape == y_gt.shape

    prec, rec = np.ones((n_classes, n_classes)) * -1.0, np.ones((n_classes, n_classes)) * -1.0
    maps = np.ones((n_classes, n_classes)) * -1.0
    n_occurs = np.zeros((n_classes, n_classes))

    for c_j in range(n_classes):
        y_gt_sub = y_gt[y_gt_mask[:, c_j] == 1]  # contains the subset of samples where c_j exists
        y_pred_sub = y_pred[y_gt_mask[:, c_j] == 1]

        pr_j, re_j, f1_j, n_j = precision_recall_fscore_support(y_gt_sub, (y_pred_sub >= thresh).astype(np.uint8),
                                                                average=None)
        map_j = average_precision_score(y_gt_sub, y_pred_sub, average=None)

        for c_i in range(n_classes):
            if c_i == c_j:
                continue
            if n_j[c_i] == 0:
                continue
            n_occurs[c_i, c_j] = n_j[c_i]
            if np.isnan(pr_j[c_i]) or np.isnan(re_j[c_i]):
                prec[c_i, c_j] = 0
                rec[c_i, c_j] = 0
            else:
                prec[c_i, c_j] = pr_j[c_i]
                rec[c_i, c_j] = re_j[c_i]

            if np.isnan(map_j[c_i]):
                maps[c_i, c_j] = 0
            else:
                maps[c_i, c_j] = map_j[c_i]

    if avg:
        return np.mean(prec[prec != -1]), np.mean(rec[rec != -1]), n_occurs, np.mean(maps[maps != -1])
    else:
        return prec, rec, n_occurs, maps


def avg_scores(score):
    return np.mean(score[score >= 0]) * 100


def get_f1(prec, rec):
    return 2 * (prec * rec) / (prec + rec + 1e-9)


def standard_metric(y_pred, y_gt, thresh=0.5):
    y_pred = np.concatenate(y_pred, 0)

    y_gt = np.concatenate(y_gt, 0)

    pr, re, _, n = precision_recall_fscore_support(y_gt, (y_pred >= thresh).astype(np.uint8), average=None)
    maps = average_precision_score(y_gt, y_pred, average=None)

    return pr, re, n, maps


def conditional_metric(y_pred, y_gt, t=0, thresh=0.5, avg=True):
    """
    The official implementation of the action-conditional metrics.
    
    y_pred is a list of un-thresholded predictions [(T1, C), (T2, C), ...]. Each element of the list is a different video, where the shape is (Time, #Classes).
    y_gt is a list of binary ground-truth labels [(T1, C), (T2, C), ...]. Each element of the list is a different video, where the shape is (Time, #Classes).
    t is an integer. If =0, measures in-timestep coocurrence. If >0, it measures conditional score of succeeding
        actions (i.e. if c_i follows c_j). If <0 it measure conditional score of preceeding actions (i.e. if c_i preceeds c_j).
    thresh is a value in range (0, 1) which binarizes the predicted probabilities
    avg determines whether it returns a single score or class-wise scores

    Returns

    prec: the action-conditional precision score
    rec: the action-conditional recall score
    n_s: the number of samples for the pair of actions. Has shape (#Classes, #Classes).
    map: the action-conditional mAP score

    """
    y_pred = np.concatenate(y_pred, 0)

    if t == 0:
        y_gt = np.concatenate(y_gt, 0).astype(np.uint8)

        return conditional_t(y_pred, y_gt, y_gt, thresh, avg)
    else:
        y_gt_mask = []
        for vid_y_gt in y_gt:

            if t > 0:  # looks at previous t time-steps
                cumsum = np.cumsum(vid_y_gt, 0)
                rolled = np.roll(cumsum, t, 0)

                rolled[:t] = 0
                n_in_last_t = cumsum - rolled
            else:  # looks at next 0-t time-steps
                vid_y_gt_flipped = np.flip(vid_y_gt, 0)

                cumsum = np.cumsum(vid_y_gt_flipped, 0)
                rolled = np.roll(cumsum, t, 0)

                rolled[:0 - t] = 0
                n_in_last_t = cumsum - rolled

                n_in_last_t = np.flip(n_in_last_t, 0)

            n_in_last_t = np.clip(n_in_last_t, 0, 1)
            masked = n_in_last_t - vid_y_gt
            # 1: present before/after, but not in current
            # 0: present before/after and in current, or not present before/after and not in current
            # -1: not present before/after and in current

            masked = np.clip(masked, 0, 1)
            y_gt_mask.append(masked)

        y_gt = np.concatenate(y_gt, 0).astype(np.uint8)
        y_gt_mask = np.concatenate(y_gt_mask, 0).astype(np.uint8)

        return conditional_t(y_pred, y_gt, y_gt_mask, thresh, avg)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode)


def sampled_25(probs, labels, mask):
    """
    Approximate version of the charades evaluation function
    """
    valid_t = int(sum(mask))
    p1_ = probs[:valid_t, :]
    l1_ = labels[:valid_t, :]
    sc = valid_t / 25.
    p1 = p1_[1::int(sc), :][:25, :]
    l1 = l1_[1::int(sc), :][:25, :]
    return p1, l1 


def sampled_25_inference(probs, labels, apm):
    """
    Approximate version of the charades evaluation function
    """
    valid_t = probs.shape[0]
    if valid_t>25:
        p1_ = probs[:valid_t, :]
        l1_ = labels[:valid_t, :]
        sc = valid_t / 25.
        p1 = p1_[1::int(sc), :][:25, :]
        l1 = l1_[1::int(sc), :][:25, :]
        apm.add(p1, l1)


def mask_probs(probs, mask):
    valid_t = int(sum(mask))
    p1_ = probs[:valid_t, :]
    return p1_ 


def focal_loss(preds, targets):
  ''' 
  Action focal loss.
  '''
  targets=targets.transpose(1,2)
  pos_inds = targets.eq(1).float()
  neg_inds = targets.lt(1).float()

  neg_weights = torch.pow(1 - targets, 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss / len(preds)


def slide_win(features, labels, slide_rate,num_clips):
    features_to_append = torch.zeros(slide_rate - len(features) % slide_rate, features.shape[-1])
    labels_to_append = torch.zeros(slide_rate - len(labels) % slide_rate, labels.shape[-1])
    features = torch.cat((features, features_to_append), 0)
    labels = torch.cat((labels, labels_to_append), 0)
    features = torch.stack(
        [features[i:i + int(num_clips)] for i in range(0, len(features) - int(num_clips) + 1, slide_rate)])
    labels = torch.stack([labels[i:i + int(num_clips)] for i in range(0, len(labels) - int(num_clips) + 1, slide_rate)])
    return features, labels


def gaussian1D(ss, sigma=1):
  m = (ss - 1.) / 2.
  x = np.ogrid[-m:m + 1]
  h = np.exp(-(x * x ) / (sigma * sigma))
  h[h < np.finfo(h.dtype).eps * h.max()] = 0
  return h


def generate_gaussian(heatmap, center, radius, tau=3, k=1):
  diameter = (2 * radius + 1)
  gaussian = gaussian1D(diameter, sigma=diameter/tau)
  t = int(center)
  T = heatmap.shape[0]
  left, right = min(t, radius), min(T - t, radius + 1)
  masked_heatmap = heatmap[t - left:t + right]
  masked_gaussian = gaussian[radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap


def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')