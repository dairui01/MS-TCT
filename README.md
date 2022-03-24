# MS-TCT implementation [CVPR2022] 

In this repository, we provide an implementation of MS-TCT on Charades (Localization setting, i.e., Charades_v1_localize). 
If you want to train and evaluate MS-TCT, you can follow the following steps. 

## Prepare the I3D feature
Like the previous works (e.g., MLAD, PDAN), MS_TCT is built on top of the pre-trained I3D features. Thus, feature extraction is needed before training the network.  
1. Please download the Charades dataset from this [link](https://prior.allenai.org/projects/charades).
2. Follow this [repository](https://github.com/piergiaj/pytorch-i3d) to extract the snippet-level I3D feature.

## Dependencies 
Please satisfy the following dependencies to train MS-TCT correctly: 
- pytorch 1.9
- python 3.8 
- timm 0.4.12
- pickle5
- scikit-learn
- numpy

## Quick Start
1. Change the _rgb_root_ to the extracted feature path in the _train.py_. 
2. Use `./run_MSTCT_Charades.sh` for training on Charades-RGB. The best logits will be saved automatically in _./save_logit_. 
3. Use `python Evaluation.py -pkl_path /best_logit_path/` to evaluate the model with the per-frame mAP and the  action-conditional metrics.

## Remarks
- The network implementation is in ./MSTCT/ folder. 
- RGB and Optical flow are following the same training process. Both modalities can be added in the logit-level to have the two-stream performance. 
- As we set clip length to 256, thus the sliding window operation is not needed to Charades, i.e., the longest video at inference time is smaller than 2048 frames. 
- In practice, we trained MS-TCT with a Tesla V100 GPU to shrink the computation time. But as MS-TCT is not large, GTX 1080 Ti can be sufficient for running the network. 
- We will release the MS-TCT code along with the pre-trained models. 
