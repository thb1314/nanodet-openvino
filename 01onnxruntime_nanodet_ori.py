import onnxruntime as rt
import numpy as np
import time
import onnx
import cv2
import torch
from torch._C import dtype
from torchvision.ops import nms as torch_nms

def _normalize(img, mean, std):
    img = img.astype(np.float32) / 255
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img

def preprocess(img):
    img = cv2.resize(img, dsize=(320,320))
    mean,std = [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
    img = _normalize(img, mean, std)
    img = np.transpose(img, axes=[2, 0, 1])
    img = np.expand_dims(img, 0)
    return img


def get_onnx_runner(onnx_filepath):
    sess = rt.InferenceSession(onnx_filepath)
    input_names = [item.name for item in sess.get_inputs()]
    label_names = [item.name for item in sess.get_outputs()]
    def runner(input_tensors):
        nonlocal label_names
        nonlocal input_names
        pred_onnx = sess.run(label_names, dict(zip(input_names, input_tensors)))
        return dict(zip(label_names,pred_onnx))
    return runner

def softmax(x, axis=None):
    axis = axis or 0
    x_max = x.max(axis = axis, keepdims = True)
    return np.exp(x- x_max) / np.sum(np.exp(x - x_max), axis=axis, keepdims=True)

def dispred2bbox(dfl_det, label, score, x, y, stride):
    ct_x = (x + 0.5) * stride
    ct_y = (y + 0.5) * stride
    dfl_det = dfl_det.reshape(4, 8)
    dfl_det_softmax = softmax(dfl_det, axis=1)
    
    dis = dfl_det_softmax @ np.arange(0, 8).astype(np.float32)
    dis_pred = dis.reshape(-1) * stride
    x_min = max(0, ct_x - dis_pred[0])
    y_min = max(0, ct_y - dis_pred[1])
    x_max = min(320, ct_x + dis_pred[2])
    y_max = min(320, ct_y + dis_pred[3])
    return (x_min, y_min, x_max, y_max, score, label)


def nms(ori_dets):
    tmp_arr = np.array(ori_dets, dtype=np.float32)
    dets, score = tmp_arr[:,:4], tmp_arr[:,4]
    dets_tensor = torch.as_tensor(dets)
    score_tensor = torch.as_tensor(score)
    keep_index = torch_nms(dets_tensor, score_tensor, 0.5)
    new_dets = [ori_dets[index] for index in keep_index]
    return new_dets

def decode_infer(cls_pred, dis_pred, stride, score_threshold):
    feature_h, feature_w = 320 // stride, 320 // stride
    ret_dict = dict()

    for idx in range(feature_h * feature_w):
        row = idx // feature_w
        col = idx % feature_w
        
        scores = cls_pred[idx]
        max_score_index = scores.argmax()
        max_score = scores[max_score_index]
        if(max_score > score_threshold):
            bbox_pred = dis_pred[idx]
            ret_list = ret_dict.setdefault(max_score_index, list())
            ret_list.append(
                dispred2bbox(bbox_pred, max_score_index, max_score, col, row, stride)
            )
    for k,v in ret_dict.items():
        ret_dict[k] = nms(v)
    return ret_dict

if __name__ == "__main__":
    runner = get_onnx_runner("bak/nanodet-clean.onnx")
    ori_image = cv2.imread('car.jpg')
    img = preprocess(ori_image)
    # ori_image = cv2.resize(ori_image, dsize=(320,320))
    outs = runner([img, ])
    heads_info = {
        "cls_pred_stride_8": ("dis_pred_stride_8", 8),
        "cls_pred_stride_16": ("dis_pred_stride_16", 16),
        "cls_pred_stride_32" : ("dis_pred_stride_32", 32),
    }
    score_threshold = 0.4
    ret_dict = dict()
    assert(1 == img.shape[0])
    for b in range(img.shape[0]):

        for cls_name in heads_info:
            dis_name, stride = heads_info[cls_name]
            cls_pred = outs[cls_name][b]
            dis_pred = outs[dis_name][b]
            ret_dict.update(decode_infer(cls_pred, dis_pred, stride, score_threshold))
    labels = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                    "hair drier", "toothbrush")
    color_list = (
        (216 , 82 , 24),
        (236 ,176 , 31),
        (125 , 46 ,141),
        (118 ,171 , 47),
        ( 76 ,189 ,237),
        (238 , 19 , 46),
        ( 76 , 76 , 76),
        (153 ,153 ,153),
        (255 ,  0 ,  0),
        (255 ,127 ,  0),
        (190 ,190 ,  0),
        (  0 ,255 ,  0),
        (  0 ,  0 ,255),
        (170 ,  0 ,255),
        ( 84 , 84 ,  0),
        ( 84 ,170 ,  0),
        ( 84 ,255 ,  0),
        (170 , 84 ,  0),
        (170 ,170 ,  0),
        (170 ,255 ,  0),
        (255 , 84 ,  0),
        (255 ,170 ,  0),
        (255 ,255 ,  0),
        (  0 , 84 ,127),
        (  0 ,170 ,127),
        (  0 ,255 ,127),
        ( 84 ,  0 ,127),
        ( 84 , 84 ,127),
        ( 84 ,170 ,127),
        ( 84 ,255 ,127),
        (170 ,  0 ,127),
        (170 , 84 ,127),
        (170 ,170 ,127),
        (170 ,255 ,127),
        (255 ,  0 ,127),
        (255 , 84 ,127),
        (255 ,170 ,127),
        (255 ,255 ,127),
        (  0 , 84 ,255),
        (  0 ,170 ,255),
        (  0 ,255 ,255),
        ( 84 ,  0 ,255),
        ( 84 , 84 ,255),
        ( 84 ,170 ,255),
        ( 84 ,255 ,255),
        (170 ,  0 ,255),
        (170 , 84 ,255),
        (170 ,170 ,255),
        (170 ,255 ,255),
        (255 ,  0 ,255),
        (255 , 84 ,255),
        (255 ,170 ,255),
        ( 42 ,  0 ,  0),
        ( 84 ,  0 ,  0),
        (127 ,  0 ,  0),
        (170 ,  0 ,  0),
        (212 ,  0 ,  0),
        (255 ,  0 ,  0),
        (  0 , 42 ,  0),
        (  0 , 84 ,  0),
        (  0 ,127 ,  0),
        (  0 ,170 ,  0),
        (  0 ,212 ,  0),
        (  0 ,255 ,  0),
        (  0 ,  0 , 42),
        (  0 ,  0 , 84),
        (  0 ,  0 ,127),
        (  0 ,  0 ,170),
        (  0 ,  0 ,212),
        (  0 ,  0 ,255),
        (  0 ,  0 ,  0),
        ( 36 , 36 , 36),
        ( 72 , 72 , 72),
        (109 ,109 ,109),
        (145 ,145 ,145),
        (182 ,182 ,182),
        (218 ,218 ,218),
        (  0 ,113 ,188),
        ( 80 ,182 ,188),
        (127 ,127 ,  0),
    )

    for v in ret_dict.values():
        for left, top, right, bottom, score, label_index in v:
            new_left = int(left / 320 * ori_image.shape[1])
            new_top = int(top / 320 * ori_image.shape[0])
            new_right = int(right / 320 * ori_image.shape[1])
            new_bottom = int(bottom / 320 * ori_image.shape[0])
            cv2.rectangle(ori_image, (new_left, new_top), (new_right, new_bottom), color = color_list[label_index])

            # 写字
            ret, baseline = cv2.getTextSize( labels[label_index], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            
            cv2.rectangle(ori_image, (new_left, new_top - ret[1] - baseline),
                         (new_left + ret[0], new_top), color_list[label_index], -1)
            cv2.putText(ori_image, labels[label_index], (new_left, new_top - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('image',ori_image)
    cv2.waitKey(0)


    