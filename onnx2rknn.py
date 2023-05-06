import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import argparse


CLASS_SET = ["person"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# def process(input, mask, anchors):
def process(input, mask, anchors, img_sz):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(img_sz/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= obj_thresh)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= obj_thresh)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores, nms_thresh):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

# def yolov5_post_process(input_data):
def yolov5_post_process(input_data, img_sz, obj_thresh, nms_thresh):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors, img_sz)
        b, c, s = filter_boxes(b, c, s, obj_thresh)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, nms_thresh)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, ratio, padding, boxes, scores, classes, CLASS_SET):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        TL_X, TL_Y, BR_X, BR_Y = box
        print('class: {}, score: {}'.format(CLASS_SET[cl], score))
        print('box coordinate topleft: ({}, {}), bottomright: ({}, {})'.format(
            TL_X, TL_Y, BR_X, BR_Y))

        TL_X = int((TL_X - padding[0]) / ratio[0])
        TL_Y = int((TL_Y - padding[1]) / ratio[1])
        BR_X = int((BR_X - padding[0]) / ratio[0])
        BR_Y = int((BR_Y - padding[1]) / ratio[1])

        cv2.rectangle(image, (TL_X, TL_Y), (BR_X, BR_Y), (255, 0, 0), 1)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASS_SET[cl], score),
                    (TL_X, TL_Y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=[320], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='NMS IoU threshold')
    parser.add_argument('--quantize', action='store_true',
                        help='quantize on is True')
    parser.add_argument('--onnx-model', type=str, default='best.onnx',
                        help='onnx which needed to convert to rknn model')
    parser.add_argument('--rknn-model', type=str,
                        default='best.rknn', help='exported rknn model')
    parser.add_argument('--target-platform', type=str,
                        default='rk3588', help='export target platform')
    parser.add_argument('--dataset', type=str, default='./dataset.txt',
                        help='dataset store the image path')
    parser.add_argument('--test-img', type=str, default='test.jpg', help='test image')
    opt = parser.parse_args()
    return opt



def run(
        img_size=320,
        conf_thres= 0.25,
        iou_thres=0.45,
        quantize=True,
        onnx_model='best.onnx',
        rknn_model='best.rknn',
        target_platform='rk3588',
        dataset='./dataset.txt',
        test_img='test.jpg'
    ):
    

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    # rknn.config(mean_values=[[0, 0, 0]], std_values=[
    #             [255, 255, 255]], target_platform='rk3588')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[
                [255, 255, 255]], target_platform=target_platform)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    # ret = rknn.load_onnx(model=ONNX_MODEL)
    ret = rknn.load_onnx(model=onnx_model)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    # ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    ret = rknn.build(do_quantization=quantize, dataset=dataset)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    # ret = rknn.export_rknn(RKNN_MODEL)
    ret = rknn.export_rknn(rknn_model)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target_platform='rk3588')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(test_img)
    img_show = img.copy()
    
    img, ratio, (dw, dh) = letterbox(img, new_shape=(img_size, img_size))
    print(ratio, dw, dh)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    np.save('./onnx_yolov5_0.npy', outputs[0])
    np.save('./onnx_yolov5_1.npy', outputs[1])
    np.save('./onnx_yolov5_2.npy', outputs[2])
    print('done')

    # post process
    input0_data = outputs[0]
    input1_data = outputs[1]
    input2_data = outputs[2]

    input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
    input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
    input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    boxes, classes, scores = yolov5_post_process(
        input_data, img_size, conf_thres, iou_thres)

    if boxes is not None:
        draw(img_show, ratio, (dw, dh), boxes, scores, classes, CLASS_SET)
    # show output
    cv2.imshow("copy", img_show)
    cv2.waitKey(0)
    cv2.imwrite("result.jpg", img_show)
    cv2.destroyAllWindows()

    rknn.release()
    


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    
