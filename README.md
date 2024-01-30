<p align="center">
  <h1 align="center">NeXt-Stereo: Focusing on Cost Aggregation and Disparity Refinement for Lightweight Stereo-Matching</h1>
  <p align="center">


# How to use

## Environment
* Python 3.8
* Pytorch 1.10

## Install

### Create a virtual environment and activate it.

```
conda create -n fast_acv python=3.8
conda activate fast_acv
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
Use the following command to train Fast-ACVNet+ or Fast-ACVNet on Scene Flow

Firstly, train network for 24 epochs,
```
python main_sceneflow.py --attention_weights_only True --logdir ./checkpoints/sceneflow/complete
```

Use the following command to train model on KITTI (using pretrained model on Scene Flow),
```
python main_kitti.py --loadckpt ./checkpoints/sceneflow/complete/checkpoint_000023.ckpt --logdir ./checkpoints/kitti
```

## Submitted to KITTI benchmarks
```
python save_disp.py
```


