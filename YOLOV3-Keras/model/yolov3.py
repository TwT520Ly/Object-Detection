"""
    @Author: Tan.wt
    @Time: 2018/05/25
    @File: yolov3.py
    @License: Apache License
"""

import numpy as np
import keras.backend as K
from keras.models import load_model

class YOLO:
    def __init__(self, obj_threshold, nms_threshold):
        self.t1 = obj_threshold
        self.t2 = nms_threshold
        self._yolo = load_model('')

    def nms_boxes(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        '''
        使用非极大值抑制，计算剩余的box，返回值为原先列表的下标
        :param boxes: [x, y, w, h] * nums
        :param scores: score * nums
        :return: the index
        '''
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        area = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            over = inter / (area[i] + area[order[1:]] - inter)
            indexs = np.where(over <= self.t2)[0]
            order = order[indexs + 1]

        keep = np.append(keep)

        return keep

    def filter_object_boxes(self, boxes: np.ndarray, box_confidences: np.ndarray, box_class_probs: np.ndarray):

        box_scores = box_confidences * box_class_probs
        # get the max score of all the classes -> return index
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        # filter
        pos = np.where(box_scores >= self.t1)

        boxes = boxes[pos]
        box_scores = box_scores[pos]
        box_class_scores = box_scores[pos]

        return boxes, box_scores, box_class_scores

    #
    # def process_feature(self):
    #
    #
    # def yolo_out(self):
    #
    # def predict(self):
