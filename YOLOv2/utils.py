
import os, glob
import cv2

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET

from box_utils import *
from Define import *

def get_dataset(xml_dirs):
    xml_paths = []
    for xml_dir in xml_dirs:
        xml_paths += glob.glob(xml_dir + '*')
    return xml_paths

def Precision_Recall(gt_boxes, gt_classes, pred_boxes, pred_classes, threshold_iou = 0.5):
    recall = 0.0
    precision = 0.0

    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return 1.0, 1.0
        else:
            return 0.0, 0.0

    if len(pred_boxes) != 0:
        gt_boxes_cnt = len(gt_boxes)
        pred_boxes_cnt = len(pred_boxes)

        recall_vector = np.zeros(gt_boxes_cnt)
        precision_vector = np.zeros(pred_boxes_cnt)

        for gt_index in range(gt_boxes_cnt):
            for pred_index in range(pred_boxes_cnt):
                if IOU(pred_boxes[pred_index], gt_boxes[gt_index]) >= threshold_iou:
                    recall_vector[gt_index] = True
                    if gt_classes[gt_index] == pred_classes[pred_index]:
                        precision_vector[pred_index] = True

        recall = np.sum(recall_vector) / gt_boxes_cnt
        precision = np.sum(precision_vector) / pred_boxes_cnt

    return precision, recall

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def xml_read(xml_path, type = 'xywh'):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_path = xml_path[:-4] + '.jpg'
    image_path = image_path.replace('/Annotations', '/JPEGImages')

    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)

    bboxes = []
    classes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        if not label in CLASS_NAMES:
            continue

        classes.append(LABEL_DIC[label])

        bbox = obj.find('bndbox')
        
        bbox_xmin = int(bbox.find('xmin').text.split('.')[0])
        bbox_xmax = int(bbox.find('xmax').text.split('.')[0])
        bbox_ymin = int(bbox.find('ymin').text.split('.')[0])
        bbox_ymax = int(bbox.find('ymax').text.split('.')[0])

        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin

        if type == 'xywh':
            bboxes.append((bbox_xmin, bbox_ymin, bbox_width, bbox_height))
        else:
            bboxes.append((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
    
    return image_path, bboxes, classes

def read_xmls(xml_paths):
    img_paths = []
    rois = []
    classes = []

    for xml_path in xml_paths:
        img_path, _rois, _classes = xml_read(xml_path, 'xywh')

        img_paths.append(img_path)
        rois.append(_rois)
        classes.append(_classes)

    return img_paths, rois, classes

def Encode(xml_paths):

    np_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), np.float32)
    np_label_data = np.zeros((BATCH_SIZE, GRID_H, GRID_W, N_ANCHORS, BOX_SIZE + 1), np.float32)

    xml_img_paths, xml_rois, xml_classes = read_xmls(xml_paths)

    for i, img_path, rois, classes in zip(range(len(xml_img_paths)), xml_img_paths, xml_rois, xml_classes):

        img = cv2.imread(img_path)
        if img is None:
            continue

        raw_h, raw_w, _ = img.shape
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

        img = np.asarray(img, dtype = np.float32)
        label = np.zeros([GRID_H, GRID_W, N_ANCHORS, BOX_SIZE + 1], dtype = np.float32)
            
        for roi, cls in zip(rois, classes):
            active_idxs = get_active_anchors(roi, ANCHORS)
            grid_x, grid_y = get_grid_cell(roi, raw_w, raw_h, GRID_W, GRID_H)

            for active_idx in active_idxs:
                anchor_label = roi2label(roi, ANCHORS[active_idx], raw_w, raw_h, GRID_W, GRID_H)
                label[grid_y, grid_x, active_idx] = np.concatenate((anchor_label, [cls], [1.0]))
        
        np_image_data[i] = img
        np_label_data[i] = label
            
    return np_image_data, np_label_data

def Decode_GT(image_data, label_data):
    
    img = image_data.astype(np.uint8)

    for y in range(GRID_H):
        for x in range(GRID_W):
            for z in range(N_ANCHORS):
                anchor = ANCHORS[z]
                cx, cy, w, h, cls, conf = label_data[y, x, z, :]
                
                if conf == 1:
                    w *= anchor[0]
                    h *= anchor[1]

                    w *= IMAGE_WIDTH
                    h *= IMAGE_HEIGHT

                    cx *= GRID_SIZE
                    cy *= GRID_SIZE

                    cx += (x * GRID_SIZE)
                    cy += (y * GRID_SIZE)

                    xmin = int((cx - w/2))
                    ymin = int((cy - h/2))
                    xmax = int((cx + w/2))
                    ymax = int((cy + h/2))
                    
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow('Decode', img)
    cv2.waitKey(0)

def Decode(output, object_threshold = 0.5):
    
    bboxes = []
    classes = []
    
    for y in range(GRID_H):
        for x in range(GRID_W):
            for z in range(N_ANCHORS):
                anchor = ANCHORS[z]

                cx, cy, w, h = output[y, x, z, :4]
                conf = sigmoid(output[y, x, z, 4])
                cls = output[y, x, z, 5:]

                if conf >= object_threshold:
                    cx = sigmoid(cx)
                    cy = sigmoid(cy)

                    w = np.exp(w)
                    h = np.exp(h)
                    
                    w *= anchor[0]
                    h *= anchor[1]

                    w *= IMAGE_WIDTH
                    h *= IMAGE_HEIGHT

                    cx *= GRID_SIZE
                    cy *= GRID_SIZE

                    cx += (x * GRID_SIZE)
                    cy += (y * GRID_SIZE)

                    xmin = max(min(int((cx - w/2)), IMAGE_WIDTH - 1), 0)
                    ymin = max(min(int((cy - h/2)), IMAGE_HEIGHT - 1), 0)
                    xmax = max(min(int((cx + w/2)), IMAGE_WIDTH - 1), 0)
                    ymax = max(min(int((cy + h/2)), IMAGE_HEIGHT - 1), 0)

                    bboxes.append([xmin, ymin, xmax, ymax])
                    classes.append(np.argmax(cls))

    return bboxes, classes
