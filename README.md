# [TGRS2023] SmallTrack: Wavelet Pooling and Graph Enhanced Classification for UAV Small Object Tracking 

This is an official pytorch implementation of the 2023 IEEE Transactions on Geoscience and Remote Sensing paper: 
```
SmallTrack: Wavelet Pooling and Graph Enhanced Classification for UAV Small Object Tracking
(accepted by IEEE Transactions on Geoscience and Remote Sensing, DOI: 10.1109/TGRS.2023.3305728)
```

![image](https://github.com/xyl-507/SmallTrack/blob/main/figs/fig%202.jpg)

The paper can be downloaded from [IEEE Xplore](https://ieeexplore.ieee.org/document/10220112) and [Researchgate](https://www.researchgate.net/publication/373148569_SmallTrack_Wavelet_Pooling_and_Graph_Enhanced_Classification_for_UAV_Small_Object_Tracking)

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

Specially for: ImportError: cannot import name 'region' from partially initialized module 'toolkit.utils'
```
Solution is：commented out 'from toolkit.utils.region import vot_overlap, vot_float2str' in /public/workspace/xyl/uavtrackers/SmallTrack-main/tools/test.py
```
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

The dataset annotated at intervals of 10 frames (VTUAV) needs to be modified: 
- 1）The GT is read at intervals of 10 frames: line80 in '/public/workspace/xyl/uavtrackers/SmallTrack-main/toolkit/datasets/video.py'
- 2）The intervals of VTUAV are Spaces：line.split(' ') in '/public/workspace/xyl/uavtrackers/SmallTrack-main/make_json-VTUAV.py'
Json file is in [GitHub](https://github.com/xyl-507/SmallTrack/releases/tag/Json)

### Test tracker
- Note that it is not necessary to generate the json files for the test dataset as per pysot.
- We read the dataset format online to generate the corresponding dictionary, the relevant files are in [visdrone.py](https://github.com/xyl-507/SmallTrack/blob/main/toolkit/datasets/visdrone.py)

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

- training datasets and corresponding Json files are provided in [[Baidu Drive]](https://pan.baidu.com/s/1-20iGnDMT3ewtA2uNsGa6g?pwd=vrw4) for COCO, DET, GOT-10k, LaSOT, VID, YouTube-BoundingBoxes
- See [TRAIN.md](TRAIN.md) for detailed instruction.


### Acknowledgement
The code based on the [PySOT](https://github.com/STVIR/pysot) , [SiamBAN](https://github.com/hqucv/siamban) ,
[GAL](https://ieeexplore.ieee.org/document/9547682/) , [WaveCNets](https://ieeexplore.ieee.org/document/9508165/) and [Wavelet-Attention](https://link.springer.com/article/10.1007/s00530-022-00889-8)
We would like to express our sincere thanks to the contributors.

### Citation:
If you find this work useful for your research, please cite the following papers:
```
@ARTICLE{10220112,
  author={Xue, Yuanliang and Jin, Guodong and Shen, Tao and Tan, Lining and Wang, Nian and Gao, Jing and Wang, Lianfeng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SmallTrack: Wavelet Pooling and Graph Enhanced Classification for UAV Small Object Tracking}, 
  year={2023},
  volume={61},
  pages={1-15},
  keywords={Target tracking;Object tracking;Wavelet transforms;Feature extraction;Task analysis;Remote sensing;Visualization;Aerial tracking;graph enhanced classification;remote sensing;Siamese neural network;wavelet pooling layer (WPL)},
  doi={10.1109/TGRS.2023.3305728}}
```
If you have any questions about this work, please contact with me via xyl_507@outlook.com