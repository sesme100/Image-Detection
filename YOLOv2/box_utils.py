
import numpy as np

from Define import *

def convert_bboxes_scale(bboxes, original_shape, change_shape):
    for i in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[i]

        xmin = int(xmin / original_shape[1] * change_shape[1])
        ymin = int(ymin / original_shape[0] * change_shape[0])
        xmax = int(xmax / original_shape[1] * change_shape[1])
        ymax = int(ymax / original_shape[0] * change_shape[0])

        bboxes[i] = [xmin, ymin, xmax, ymax]
    return bboxes

def IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = (xB - xA) * (yB - yA)

    if (xB - xA) < 0.0:
        return 0.0

    if (yB - yA) < 0.0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if interArea < 0.0:
        return 0.0

    if (boxAArea + boxBArea - interArea) <= 0.0:
        return 0.0

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return max(iou, 0.0)


def iou_wh(r1, r2):

	min_w = min(r1[0],r2[0])
	min_h = min(r1[1],r2[1])
	area_r1 = r1[0]*r1[1]
	area_r2 = r2[0]*r2[1]
		
	intersect = min_w * min_h		
	union = area_r1 + area_r2 - intersect

	return intersect/union

def get_grid_cell(roi, raw_w, raw_h, grid_w, grid_h):

	x_center = roi[0] + roi[2]/2.0
	y_center = roi[1] + roi[3]/2.0

	grid_x = int(x_center/float(raw_w)*float(grid_w))
	grid_y = int(y_center/float(raw_h)*float(grid_h))
		
	return grid_x, grid_y

def get_active_anchors(roi, anchors):
	 
	indxs = []
	iou_max, index_max = 0, 0
	for i,a in enumerate(anchors):
		iou = iou_wh(roi[2:], a)
		if iou>IOU_TH:
			indxs.append(i)
		if iou > iou_max:
			iou_max, index_max = iou, i

	if len(indxs) == 0:
		indxs.append(index_max)

	return indxs

def roi2label(roi, anchor, raw_w, raw_h, grid_w, grid_h):
	
	x_center = roi[0]+roi[2]/2.0
	y_center = roi[1]+roi[3]/2.0

	grid_x = x_center/float(raw_w)*float(grid_w)
	grid_y = y_center/float(raw_h)*float(grid_h)
	
	grid_x_offset = grid_x - int(grid_x)
	grid_y_offset = grid_y - int(grid_y)

	roi_w_scale = roi[2]/float(raw_w)/anchor[0]
	roi_h_scale = roi[3]/float(raw_h)/anchor[1]

	label=[grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale]
	
	return label
