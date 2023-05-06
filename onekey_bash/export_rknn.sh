#!/bin/bash
source config_onekey.sh

# 切换到pytorch虚拟环境
source activate pytorch

# 在pytorch 环境下导出 onnx
python export.py  ${__WEIGHTS} ${__IMG_SZ} --batch-size 1 --include torchscript onnx --optimize --simplify --opset 12
# parser = argparse.ArgumentParser()
# parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
# parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image (height, width)')
# parser.add_argument('--batch-size', type=int, default=1, help='batch size')
# parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx', 'coreml'], help='include formats')
# parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
# parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
# parser.add_argument('--train', action='store_true', help='model.train() mode')
# parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
# parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
# parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
# parser.add_argument('--opset', type=int, default=13, help='ONNX: opset version')

# 将 onnx 模型移动到  export 文件夹下
mv ${__WEIGHTSPATH}/${__WEIGHTSNAME}.onnx ./export/${__PRJNAME}.onnx

# 切换 rknn 虚拟环境
source activate rknn

# 将 onnx 模型转换为 rknn 模型
python onnx2rknn.py ${__IMG_SZ} ${__ONNX_MODEL} ${__RKNN_MODEL} ${__TARGET_PLATFORM} ${__DATA_SET} ${__TEST_IMG} ${__QUANTIZE}
# parser.add_argument('--img-size', type=int, default=[320], help='inference size h,w')
# parser.add_argument('--conf-thres', type=float,
#                     default=0.25, help='confidence threshold')
# parser.add_argument('--iou-thres', type=float,
#                     default=0.45, help='NMS IoU threshold')
# parser.add_argument('--quantize', action='store_true',
#                     help='quantize on is True')
# parser.add_argument('--onnx-model', type=str, default='best.onnx',
#                     help='onnx which needed to convert to rknn model')
# parser.add_argument('--rknn-model', type=str,
#                     default='best.rknn', help='exported rknn model')
# parser.add_argument('--target-platform', type=str,
#                     default='rk3588', help='export target platform')
# parser.add_argument('--dataset', type=str, default='./dataset.txt',
#                     help='dataset store the image path')
# parser.add_argument('--test-img', type=str, default='test.jpg', help='test image')


# 移除临时文件
rm *.npy
