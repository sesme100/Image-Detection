
import os, cv2, glob
from time import time

import random
import numpy as np
import tensorflow as tf

from YOLOv2 import *

from Loss import *
from Define import *

from utils import *
from box_utils import *

if __name__ == '__main__':
    
    train_xml_paths = get_dataset(TRAIN_DB_XML_DIRS)
    test_xml_paths = get_dataset(TEST_DB_XML_DIRS)
    
    print('# Train Data : {}'.format(len(train_xml_paths)))
    print('# Test Data : {}'.format(len(test_xml_paths)))
    
    model_path_format = './model/yolov2_{}.ckpt'
    
    input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], name = 'input')
    label_var = tf.placeholder(tf.float32, [None, GRID_H, GRID_W, N_ANCHORS, BOX_SIZE + 1], name = 'label')
    training_var = tf.placeholder(tf.bool)

    learning_rate_var = tf.placeholder(tf.float32, name = 'learning_rate')
    tf.summary.scalar('learning_rate', learning_rate_var)
    
    with tf.variable_scope('network'):
        yolov2 = YOLOv2(input_var, training_var)
        print(yolov2)
        
    with tf.name_scope('loss'):
        loss_op = yolo_loss(yolov2, label_var)
        print(loss_op)
        
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate = learning_rate_var).minimize(loss_op)
        #train_op = tf.train.MomentumOptimizer(learning_rate = learning_rate_var, momentum = 0.9).minimize(loss_op)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # tensorboard
        merged_summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(TENSORBOARD_DIR, graph = sess.graph)

        restore_index = 0
        if restore_index != 0:
            saver.restore(sess, model_path.format(restore_index))
            
        if False:
            gd = sess.graph.as_graph_def()
            converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ['network/outputs'])
            tf.train.write_graph(converted_graph_def, './', 'YOLOv2_VOC.pb', as_text=False)
            input('freeze graph save complete')
            
        train_start_time = time()
        
        iter_losses = []
        iter_learning_rate = 1e-4

        for iter in range(1 + restore_index, MAX_ITERS + 1):
            if iter == int(MAX_ITERS * 0.5) or iter == int(MAX_ITERS * 0.75) or iter == int(MAX_ITERS * 0.9):
                iter_learning_rate /= 10

            random.shuffle(train_xml_paths)

            encode_image_data, encode_label_data = Encode(train_xml_paths[:BATCH_SIZE])

            # Decode
            #for i in range(BATCH_SIZE):
            #    Decode_GT(encode_image_data[i], encode_label_data[i])

            _, summary, iter_loss = sess.run([train_op, merged_summary_op, loss_op], feed_dict = {
                                                                                    input_var : encode_image_data,
                                                                                    label_var : encode_label_data,
                                                                                    learning_rate_var : iter_learning_rate,
                                                                                    training_var : True
                                                                                    })
            train_writer.add_summary(summary, iter)
            iter_losses.append(iter_loss)

            if iter % LOG_ITER == 0:
                log_time = time() - train_start_time

                print('iter : %d, time : %.2f, loss : %.4f'%(iter, log_time, np.mean(iter_losses)))

                train_start_time = time()
                iter_losses = []

            if iter % SAVE_ITER == 0 or iter > 10000:
                saver.save(sess, model_path_format.format(iter))

                #test mAP
                test_start_time = time()

                precisions = []
                recalls = []

                for test_xml_path in test_xml_paths:

                    image_path, gt_bboxes, gt_classes = xml_read(test_xml_path, 'xyxy')
                    img = cv2.imread(image_path)
                    if img is None:
                        continue

                    original_shape = img.shape
                    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
                    resize_shape = img.shape
                    
                    preds = sess.run(yolov2, feed_dict={ input_var : [img],
                                                         training_var : False })

                    gt_bboxes = convert_bboxes_scale(gt_bboxes, original_shape, resize_shape)
                    pred_bboxes, pred_classes = Decode(preds[0])

                    precision, recall = Precision_Recall(gt_bboxes, gt_classes, pred_bboxes, pred_classes, 0.5)
                    
                    precisions.append(precision)
                    recalls.append(recall)

                test_time = time() - test_start_time
                print('iter : %d, time : %.2f, test set Precision : %.2f, test set Recall : %.2f'%(iter, test_time, np.mean(recalls) * 100, np.mean(precisions) * 100))

                train_start_time = time()
