  <p align="center">
  <h1 align="center">NeXt-Stereo: Focusing on Cost Aggregation and Disparity Refinement for Lightweight Stereo-Matching</h1>
  <p align="center">

"### Paper Under Submission

I am currently in the process of submitting my paper to an academic journal. Once the paper is accepted, I will update this section with relevant information. Thank you for your interest and patience."

# How to use

## Environment
* Python 3.8
* Pytorch 1.10

## Install

### Create a virtual environment and activate it.

```
conda create -n nextstereo python=3.8
conda activate nextstereo
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm
pip install basicsr
```

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

## Train

Firstly, train network for 24 epochs,
```
python main_sceneflow.py --attention_weights_only True --logdir ./checkpoints/sceneflow/complete
```

Use the following command to train model on KITTI (using pretrained model on Scene Flow),
```
python main_kitti.py --loadckpt ./checkpoints/sceneflow/complete/checkpoint_000023.ckpt --logdir ./checkpoints/kitti
```


## Evaluation on Scene Flow and KITTI

### Pretrained Model
NeXt-Stereo
* [sceneflow](https://drive.google.com/file/d/1lBeCbvqwO3--5CQyFwnEear5q_EJK2zT/view?usp=sharing)

Generate disparity images of KITTI test set,
```
python save_disp.py
```

## Submitted to KITTI benchmarks
```
python save_disp.py
```


