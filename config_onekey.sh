#!/bin/bash

## train param
__PRJNAME="drp"
__PRE_TRAINED_WEIGHTSNAME="../pre-trained/yolov5s.pt"
__DATANAME="${__PRJNAME}.yaml"
__EPOCHSZ="100"
__BATCHSZ="32"
__IMSZ="320"
__NAME=" --name ${__PRJNAME} "

__PRE_TRAINED_WEIGHTS=" --weights ${__PRE_TRAINED_WEIGHTSNAME} "
__DATA=" --data data/${__DATANAME} "
__EPOCHS=" --epochs ${__EPOCHSZ} "
__BATCH_SZ=" --batch-size ${__BATCHSZ} "
__IMG_SZ=" --img-size ${__IMSZ} "
__RECT=" --rect " # 矩形训练
#__MULTI_SCALE=" --multi-scale "  # maybe wrong in small size, will change the model
#__PROJECT=" --project ${__PRJNAME}"


## predict param 
__WEIGHTSPATH="runs/train/${__PRJNAME}/weights"
__WEIGHTSNAME="best"
__SRC_IM="../test-images/test.jpg"
__SRC_VIDEO="../test-video/test.mp4"
__SRC_PATH=""
__SRC_RTSP=""
__CONFTHRESH="0.65"

__WEIGHTS=" --weights ${__WEIGHTSPATH}/${__WEIGHTSNAME}.pt "
__SOURCE=" --source  ${__SRC_IM} "
# __SOURCE=" --source  ${__SRC_VIDEO}  --view-img "
__CONF_THRESH=" --conf-thres ${__CONFTHRESH} "
__SAVE_TXT=" --save-txt "
__SAVE_CONF=" --save-conf "
__SAVE_CROP=" --save-crop "
__VISUALIZE=" --visualize "

## export param
__SIMPLIFY=" --simplify "
__INCLUD=" --include onnx " 

# __SHAPE=" inputshape=[1,3,640,640] inputshape2=[1,3,320,320] "
__SHAPE=" inputshape=[1,3,320,320] "


# export rknn
__ONNX_MODEL=" --onnx-model ./export/${__PRJNAME}.onnx "
__RKNN_MODEL=" --rknn-model ./export/${__PRJNAME}.rknn "
__TARGET_PLATFORM=" --target-platform rk3588 "
__DATA_SET=" --dataset ./export/${__PRJNAME}_dataset.txt "
__TEST_IMG=" --test-img ./export/${__PRJNAME}.png "
__QUANTIZE=" --quantize "
