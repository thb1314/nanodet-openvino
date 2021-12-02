import sys
import os
import cv2
import logging as log
import numpy as np
from openvino.inference_engine import IECore
import torch
from torchvision.ops import nms as torch_nms
import time

class MultipleOutputPostprocessor:
    def __init__(self, bboxes_layer='bboxes', scores_layer='scores', labels_layer='labels'):
        self.bboxes_layer = bboxes_layer
        self.scores_layer = scores_layer
        self.labels_layer = labels_layer

    def __call__(self, outputs):
        bboxes = outputs[self.bboxes_layer].buffer[0]
        scores = outputs[self.scores_layer].buffer[0]
        # labels = outputs[self.labels_layer].buffer[0]
        labels = [0, ] * len(scores)
        return [[0, label, score, *bbox] for label, score, bbox in zip(labels, scores, bboxes)]


def preprocess(img):
    img = cv2.resize(img, dsize=(320,320))
    img = np.transpose(img, axes=[2, 0, 1])
    img = np.expand_dims(img, 0)
    return img


def dispred2bbox(bbox_pred, label, score):
    x_min, y_min, x_max, y_max = bbox_pred.tolist()
    return (x_min, y_min, x_max, y_max, score, label)


def nms(ori_dets):
    tmp_arr = np.array(ori_dets, dtype=np.float32)
    dets, score = tmp_arr[:,:4], tmp_arr[:,4]
    dets_tensor = torch.as_tensor(dets)
    score_tensor = torch.as_tensor(score)
    keep_index = torch_nms(dets_tensor, score_tensor, 0.4)
    new_dets = [ori_dets[index] for index in keep_index]
    return new_dets

def decode_infer(cls_pred, dis_pred, score_threshold):
    ret_dict = dict()
    
    # scores = cls_pred[idx]
    scores = cls_pred
    max_score_indexes = scores.argmax(axis=-1)
    max_scores = scores[np.arange(scores.shape[0]), max_score_indexes]
    nonzero_index  = np.nonzero(max_scores > score_threshold)[0].tolist()
    if nonzero_index:
        bbox_pred = dis_pred[nonzero_index, :]
        score_indexes = max_scores[nonzero_index]
        score_indexes = score_indexes[..., None].astype(np.float32)
        max_label_indexes = max_score_indexes[nonzero_index]
        label_indexes = max_label_indexes[..., None].astype(np.float32)
        
        bbox = np.concatenate([bbox_pred, score_indexes, label_indexes], axis = -1)
        for i,index in enumerate(max_label_indexes):
            ret_list = ret_dict.setdefault(index, list())
            ret_list.append(bbox[i])
    for k,v in ret_dict.items():
        ret_dict[k] = nms(v)
    return ret_dict


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.ERROR, stream=sys.stdout)
    log.info("Creating Inference Engine...")
    ie = IECore()

    # Read IR
    log.info("Loading network")
    net = ie.read_network("./optimized/nanodet-simp.xml", "./optimized/nanodet-simp.bin")
    
    img_info_input_blob = None
    feed_dict = {}
    input_blob = "input"
    for blob_name in net.input_info:
        if len(net.input_info[blob_name].input_data.shape) == 4:
            input_blob = blob_name
        elif len(net.input_info[blob_name].input_data.shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.input_info[blob_name].input_data.shape), blob_name))

    log.info("Loading IR to the plugin...")
    input_stream = "C:/Users/THB/Videos/Gee.mp4"
    cap = cv2.VideoCapture(input_stream)
    assert cap.isOpened(), "Can't open " + str(input_stream)
    exec_net = ie.load_network(network=net, num_requests=4, device_name="CPU")
    n, c, h, w = net.input_info[input_blob].input_data.shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]

    heads_info = {
        "cls_pred_stride_8": ("dis_pred_stride_8", 8),
        "cls_pred_stride_16": ("dis_pred_stride_16", 16),
        "cls_pred_stride_32" : ("dis_pred_stride_32", 32),
    }

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

    score_threshold = 0.5

    cur_request_id = 0
    next_request_id = 1
    frame = None
    while cap.isOpened():
        ret, next_frame = cap.read()
        next_frame = cv2.resize(next_frame, fx = 0.5, fy = 0.5, dsize = None)
        if not ret:
            break  
    
        ori_image = next_frame
        img = preprocess(ori_image)
        feed_dict[input_blob] = img
        start_time = time.time()
        exec_net.start_async(request_id=next_request_id, inputs=feed_dict)

        if exec_net.requests[cur_request_id].wait(-1) == 0:
            time_span = time.time() - start_time
            print(time_span)
            start_time = time.time()
            outs = exec_net.requests[cur_request_id].output_blobs
        
            ret_dict = dict()
            
            for b in range(img.shape[0]):
                for cls_name in heads_info:
                    dis_name, stride = heads_info[cls_name]
                    # print(outs[cls_name].buffer.shape)
                    cls_pred = outs[cls_name].buffer[b]
                    dis_pred = outs[dis_name].buffer[b]
                    ret_dict.update(decode_infer(cls_pred, dis_pred, score_threshold))

            for v in ret_dict.values():
                for left, top, right, bottom, score, label_index in v:
                    label_index = int(label_index)
                    new_left = int(left / 320 * frame.shape[1])
                    new_top = int(top / 320 * frame.shape[0])
                    new_right = int(right / 320 * frame.shape[1])
                    new_bottom = int(bottom / 320 * frame.shape[0])
                    cv2.rectangle(frame, (new_left, new_top), (new_right, new_bottom), color = color_list[label_index])

                    # 写字
                    ret, baseline = cv2.getTextSize(labels[label_index], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
                    cv2.rectangle(frame, (new_left, new_top - ret[1] - baseline),
                                (new_left + ret[0], new_top), color_list[label_index], -1)
                    cv2.putText(frame, labels[label_index], (new_left, new_top - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('image',frame)
            cv2.waitKey(1)

        cur_request_id, next_request_id = next_request_id, cur_request_id
        frame = next_frame
       
if __name__ == "__main__":
    main()
