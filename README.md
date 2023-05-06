## RK3588 定制

### 训练

#### 标注数据

可以用 `cvat` 或者 `labelme` 或者其他工具对检测数据进行标注。

在 `yolov5` 同级目录新建一个 `datasets` 文件夹，将标注好的数据打包到项目名称文件夹，然后放到 `datasets` 目录下，后续其他项目也可以同样进行。


#### 修改参数

1. 修改 `./data/proj.yaml`
在 `yolov5` 文件下找到 `data` 目录，新建一个  `proj.yaml` 文件，类似于 `drp.yaml`。

修改 `proj.yaml` 中的 `nc` 和 `names`; 

`nc`: 类别数
`names`: 类别名称列表

2. 修改 `config_onekey.sh`

根据项目要求，修改 `config_onekey.sh` 文件中的相关参数；

例如: 

`__PRJNAME`, `__EPOCHSZ`, `__BATCHSZ`, `__IMSZ` 等等。

#### 开始训练

**前提条件: 已经创建了虚拟环境 `pytorch`**
**前提条件: 已经创建了虚拟环境 `pytorch`**
**前提条件: 已经创建了虚拟环境 `pytorch`**

在训练之前确定  `./models/yolo.py` 文件中的 `class Detect(nn.Module):` 类 `line:55` 里 `forward` 代码是否解注释了。

**如果训练， `forward` 应为**

```python
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
        
        ## 此处如果是训练, 请解开以下注释
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
```


**开始一键训练**
**前提条件: 已经创建了虚拟环境 `pytorch`**
**前提条件: 已经创建了虚拟环境 `pytorch`**
**前提条件: 已经创建了虚拟环境 `pytorch`**

```bash
bash onekey_bash/train.sh
```

*终端显示如下:*

```bash
(base) jxxx@desk:~/workspace/algo/yolov5_rknn$ bash onekey_bash/train.sh 
train: weights=../pre-trained/yolov5s.pt, cfg=, data=data/drp.yaml, hyp=data/hyps/hyp.scratch.yaml, epochs=100, batch_size=32, imgsz=320, rect=True, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=False, adam=False, sync_bn=False, workers=8, project=runs/train, entity=None, name=drp, exist_ok=True, quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias=latest, local_rank=-1, freeze=0, patience=100
github: ⚠️ YOLOv5 is out of date by 1203 commits. Use `git pull` or `git clone git@github.com:wangqiqi/yolov5` to update.
YOLOv5 🚀 c5360f6 torch 2.0.0+cu117 CUDA:0 (NVIDIA GeForce RTX 4090, 24209.125MB)

hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Weights & Biases: run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=1

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Focus                     [3, 32, 3]                    
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  3    156928  models.common.C3                        [128, 128, 3]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1    656896  models.common.SPP                       [512, 512, [5, 9, 13]]        
  9                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     16182  models.yolo.Detect                      [1, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
 ....
 ....
 ....

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     98/99     1.76G   0.02157  0.007322         0         3       320: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:04<00:00, 13.07it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:03<00:00,  7.54it/s]
                 all       1860       3277      0.979      0.962      0.991      0.727

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     99/99     1.76G    0.0217  0.007213         0         3       320: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:04<00:00, 12.90it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:04<00:00,  6.66it/s]
                 all       1860       3277      0.979      0.967      0.992      0.737

100 epochs completed in 0.244 hours.
Optimizer stripped from runs/train/drp/weights/last.pt, 14.3MB
Optimizer stripped from runs/train/drp/weights/best.pt, 14.3MB
Results saved to runs/train/drp
```

### 导出模型

**前提条件: 已经创建了虚拟环境 `rknn`**
**前提条件: 已经创建了虚拟环境 `rknn`**
**前提条件: 已经创建了虚拟环境 `rknn`**


在训练之前确定  `./models/yolo.py` 文件中的 `class Detect(nn.Module):` 类 `line:55` 里 `forward` 代码是否注释了。

此时应修改 forward 代码为:

```python
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
        return x # 此处如果是训练, 请注释
```

#### 修改参数

1. 修改 `config_onekey.sh` 相关参数
2. 将测试用的图片放到 `export` 文件夹下，修改 `proj_dataset.txt` 的内容为自己的图片列表

**开始一键训练**
**前提条件: 已经创建了虚拟环境 `pytorch`**
**前提条件: 已经创建了虚拟环境 `pytorch`**
**前提条件: 已经创建了虚拟环境 `pytorch`**

```bash
bash onekey_bash/export_rknn.sh
```

*终端显示如下:*

```bash
(base) jxxx@desk:~/workspace/algo/yolov5_rknn$ bash onekey_bash/export_rknn.sh 
export: weights=runs/train/drp/weights/best.pt, img_size=[320], batch_size=1, device=cpu, include=['torchscript', 'onnx'], half=False, inplace=False, train=False, optimize=True, dynamic=False, simplify=True, opset=12
YOLOv5 🚀 c5360f6 torch 2.0.0+cu117 CPU

Fusing layers... 
Model Summary: 224 layers, 7053910 parameters, 0 gradients, 16.3 GFLOPs

PyTorch: starting from runs/train/drp/weights/best.pt (14.3 MB)

TorchScript: starting export with torch 2.0.0+cu117...
TorchScript: export success, saved as runs/train/drp/weights/best.torchscript.pt (28.3 MB)

ONNX: starting export with onnx 1.13.1...
...
...
I rknn buiding done
done
--> Export rknn model
done
--> Init runtime environment
W init_runtime: Target is None, use simulator!
done
(0.3333333333333333, 0.3333333333333333) 0.0 70.0
--> Running model
Analysing : 100%|███████████████████████████████████████████████| 161/161 [00:00<00:00, 5866.87it/s]
Preparing : 100%|█████████████████████████████████████████████████| 161/161 [00:04<00:00, 39.76it/s]
W inference: The dims of input(ndarray) shape (320, 320, 3) is wrong, expect dims is 4! Try expand dims to (1, 320, 320, 3)!
W inference: The dims of input(ndarray) shape (320, 320, 3) is wrong, expect dims is 4! Try expand dims to (1, 320, 320, 3)!
W inference: The dims of input(ndarray) shape (320, 320, 3) is wrong, expect dims is 4! Try expand dims to (1, 320, 320, 3)!
W inference: The dims of input(ndarray) shape (320, 320, 3) is wrong, expect dims is 4! Try expand dims to (1, 320, 320, 3)!
done
class: person, score: 0.8913032412528992
box coordinate topleft: (159.54501259326935, 145.2426918745041), bottomright: (177.8633736371994, 186.3882895708084)
class: person, score: 0.8837643265724182
box coordinate topleft: (82.14642906188965, 183.73637676239014), bottomright: (110.66749000549316, 226.49699115753174)
class: person, score: 0.8643947243690491
box coordinate topleft: (146.98097324371338, 140.67060202360153), bottomright: (164.14344692230225, 166.86144143342972)
class: person, score: 0.8264284729957581
box coordinate topleft: (110.99540531635284, 146.92019426822662), bottomright: (130.9836596250534, 177.2033714056015)
class: person, score: 0.7860860824584961
box coordinate topleft: (146.88146114349365, 127.81136840581894), bottomright: (163.5996789932251, 148.1453576683998)
class: person, score: 0.7520475387573242
box coordinate topleft: (170.8560609817505, 123.72485554218292), bottomright: (185.6856870651245, 143.04388225078583)
class: person, score: 0.6209931373596191
box coordinate topleft: (108.76676034927368, 133.68206477165222), bottomright: (119.2998595237732, 156.40799260139465)
```

成功后，可以看到 `export` 文件下多了 `rknn` 模型。

### 部署测试模型

**TODO::**


## 原始的 README 文档

<div align="center">
<p>
<a align="left" href="https://ultralytics.com/yolov5" target="_blank">
<img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg"></a>
</p>
<br>
<div>
<a href="https://github.com/ultralytics/yolov5/actions"><img src="https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
<a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
<br>  
<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
<a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
</div>
  <br>
  <div align="center">
    <a href="https://github.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="2%"/>
    </a>
    <img width="2%" />
    <a href="https://www.linkedin.com/company/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="2%"/>
    </a>
    <img width="2%" />
    <a href="https://twitter.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="2%"/>
    </a>
    <img width="2%" />
    <a href="https://youtube.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="2%"/>
    </a>
    <img width="2%" />
    <a href="https://www.facebook.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="2%"/>
    </a>
    <img width="2%" />
    <a href="https://www.instagram.com/ultralytics/">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="2%"/>
    </a>
</div>

<br>
<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

<!-- 
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Inference</summary>

Inference with YOLOv5 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36). Models automatically download
from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>



<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading models automatically from
the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

Run commands below to reproduce results
on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on
first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the
largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).

```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>  

<details open>
<summary>Tutorials</summary>

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️
  RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Supervisely Ecosystem](https://github.com/ultralytics/yolov5/issues/2518)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [TorchScript, ONNX, CoreML Export](https://github.com/ultralytics/yolov5/issues/251) 🚀
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)

</details>

## <div align="center">Environments and Integrations</div>

Get started in seconds with our verified environments and integrations,
including [Weights & Biases](https://wandb.ai/site?utm_campaign=repo_yolo_readme) for automatic YOLOv5 experiment
logging. Click each icon below for details.

<div align="center">
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="15%"/>
    </a>
    <a href="https://www.kaggle.com/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-kaggle-small.png" width="15%"/>
    </a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-docker-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-aws-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gcp-small.png" width="15%"/>
    </a>
    <a href="https://wandb.ai/site?utm_campaign=repo_yolo_readme">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-wb-small.png" width="15%"/>
    </a>
</div>  

## <div align="center">Compete and Win</div>

We are super excited about our first-ever Ultralytics YOLOv5 🚀 EXPORT Competition with **$10,000** in cash prizes!

<p align="center">
  <a href="https://github.com/ultralytics/yolov5/discussions/3213">
  <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-export-competition.png"></a>
</p>

## <div align="center">Why YOLOv5</div>

<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure (click to expand)</summary>

<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313219-f1d70e00-9af5-11eb-9973-52b1f98d321a.png"></p>
</details>
<details>
  <summary>Figure Notes (click to expand)</summary>

* GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size
  32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS.
* EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.
* **Reproduce** by
  `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`

</details>

### Pretrained Checkpoints

[assets]: https://github.com/ultralytics/yolov5/releases

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>V100 (ms) | |params<br><sup>(M) |FLOPs<br><sup>640 (B)
|---                    |---  |---      |---      |---      |---     |---|---   |---
|[YOLOv5s][assets]      |640  |36.7     |36.7     |55.4     |**2.0** |   |7.3   |17.0
|[YOLOv5m][assets]      |640  |44.5     |44.5     |63.1     |2.7     |   |21.4  |51.3
|[YOLOv5l][assets]      |640  |48.2     |48.2     |66.9     |3.8     |   |47.0  |115.4
|[YOLOv5x][assets]      |640  |**50.4** |**50.4** |**68.8** |6.1     |   |87.7  |218.8
|                       |     |         |         |         |        |   |      |
|[YOLOv5s6][assets]     |1280 |43.3     |43.3     |61.9     |**4.3** |   |12.7  |17.4
|[YOLOv5m6][assets]     |1280 |50.5     |50.5     |68.7     |8.4     |   |35.9  |52.4
|[YOLOv5l6][assets]     |1280 |53.4     |53.4     |71.1     |12.3    |   |77.2  |117.7
|[YOLOv5x6][assets]     |1280 |**54.4** |**54.4** |**72.0** |22.4    |   |141.8 |222.9
|                       |     |         |         |         |        |   |      |
|[YOLOv5x6][assets] TTA |1280 |**55.0** |**55.0** |**72.0** |70.8    |   |-     |-

<details>
  <summary>Table Notes (click to expand)</summary>

* AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results
  denote val2017 accuracy.
* AP values are for single-model single-scale unless otherwise noted. **Reproduce mAP**
  by `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
* Speed<sub>GPU</sub> averaged over 5000 COCO val2017 images using a
  GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 instance, and
  includes FP16 inference, postprocessing and NMS. **Reproduce speed**
  by `python val.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45 --half`
* All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation).
* Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) includes reflection and scale
  augmentation. **Reproduce TTA** by `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>

## <div align="center">Contribute</div>

We love your input! We want to make contributing to YOLOv5 as easy and transparent as possible. Please see
our [Contributing Guide](CONTRIBUTING.md) to get started.

## <div align="center">Contact</div>

For issues running YOLOv5 please visit [GitHub Issues](https://github.com/ultralytics/yolov5/issues). For business or
professional support requests please visit [https://ultralytics.com/contact](https://ultralytics.com/contact).

<br>

<div align="center">
    <a href="https://github.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.linkedin.com/company/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://twitter.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://youtube.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.facebook.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.instagram.com/ultralytics/">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="3%"/>
    </a>
</div>


