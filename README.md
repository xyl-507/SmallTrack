# [TGRS2023] SmallTrack: Wavelet Pooling and Graph Enhanced Classification for UAV Small Object Tracking 

This is an official pytorch implementation of the 2023 IEEE Transactions on Geoscience and Remote Sensing paper: 
```
SmallTrack: Wavelet Pooling and Graph Enhanced Classification for UAV Small Object Tracking
(accepted by IEEE Transactions on Geoscience and Remote Sensing, DOI: 10.1109/TGRS.2023.3305728)
```

![image](https://github.com/xyl-507/SmallTrack/blob/main/figs/fig%202.jpg)

The paper can be downloaded from [IEEE Xplore](https://ieeexplore.ieee.org/document/10220112)

The models and raw results can be downloaded from [**[GitHub]**](https://github.com/xyl-507/SmallTrack/releases/tag/Downloads) or [**[BaiduYun]**](https://pan.baidu.com/s/1XRaXBpb4ab2XDrP3ViMCUw?pwd=1234). 

The tracking demos are displayed on the [Bilibili](https://www.bilibili.com/video/BV1Xv4y1e78e/)

### Proposed modules
- `DWT` in [backbone](https://github.com/xyl-507/SmallTrack/blob/main/siamban/models/backbone/resnet_atrous_DWT.py)

- `GEM` in [model_builder](https://github.com/xyl-507/SmallTrack/blob/main/siamban/models/model_builder.py)

### UAV Tracking

| Datasets | smalltrack_r50_l234|
| :--------------------: | :----------------: |
| UAV20L(Suc./Pre.) | 0.600/0.797|
| UAVDT(Suc./Pre.) | 0.637/0.866 |
| DTB70(Suc./Pre.) | 0.654/0.858 |
| VisDrone2019-SOT-test-dev(Suc./Pre.) |0.625/0.849 |
| LaTOT(Suc./Pre./N.Pre.) | 0.271/0.438/0.339 |

Note:

-  `r50_lxyz` denotes the outputs of stage x, y, and z in [ResNet-50](https://arxiv.org/abs/1512.03385).
- The suffixes `DTB70` is designed for the DTB70, the default (without suffix) is designed for UAV20L and UAVDT.
- `e20` in parentheses means checkpoint_e20.pth

## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using SmallTrack

### Add SmallTrack to your PYTHONPATH

```bash
export PYTHONPATH=/path/to/smalltrack:$PYTHONPATH
```


### demo

```bash
python tools/demo.py \
    --config experiments/smalltrack_r50_l234/config.yaml \
    --snapshot experiments/smalltrack_r50_l234/checkpoint_e20.pth
    --video demo/bag.avi
```

### Download testing datasets

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker

```bash
cd experiments/smalltrack_r50_l234
python -u ../../tools/test.py 	\
	--snapshot checkpoint_e20.pth 	\ # model path
	--dataset UAV20L 	\ # dataset name
	--config config.yaml	  # config file
```

The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker

assume still in experiments/smalltrack_r50_l234

``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset UAV20L        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'ch*'   # tracker_name
```

###  Training :wrench:

See [TRAIN.md](TRAIN.md) for detailed instruction.


### Acknowledgement
The code based on the [PySOT](https://github.com/STVIR/pysot) , [SiamBAN](https://github.com/hqucv/siamban) ,
[GAL](https://ieeexplore.ieee.org/document/9547682/) , [WaveCNets](https://ieeexplore.ieee.org/document/9508165/) and [Wavelet-Attention](https://link.springer.com/article/10.1007/s00530-022-00889-8)
We would like to express our sincere thanks to the contributors.
