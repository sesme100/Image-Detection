
import tensorflow as tf

from Define import *

def yolo_loss(preds, labels, cross_entropy = False):
    mask = labels[..., 5] #confidence
    mask = tf.cast(tf.reshape(mask, shape=(-1, GRID_H, GRID_W, N_ANCHORS)), tf.bool)
    
    labels = labels[..., :5] #x, y, w, h, class index
    
    with tf.name_scope('mask'):
        masked_label = tf.boolean_mask(labels, mask)
        pos_masked_pred = tf.boolean_mask(preds, mask)
        neg_masked_pred = tf.boolean_mask(preds, tf.logical_not(mask))

    with tf.name_scope('pred'):
        masked_pred_xy = tf.sigmoid(pos_masked_pred[..., :2])
        masked_pred_wh = tf.exp(pos_masked_pred[..., 2:4])
        masked_pred_o = tf.sigmoid(pos_masked_pred[..., 4])
        masked_pred_no_o = tf.sigmoid(neg_masked_pred[..., 4])

        if cross_entropy:
            masked_pred_c = pos_masked_pred[..., 5:]
        else:
            masked_pred_c = tf.nn.softmax(pos_masked_pred[..., 5:])
        
    with tf.name_scope('label'):
        masked_label_xy = masked_label[..., :2]
        masked_label_wh = masked_label[..., 2:4]
        masked_label_c_idx = masked_label[..., 4]
        masked_label_c = tf.reshape(tf.one_hot(tf.cast(masked_label_c_idx, tf.int32), depth=CLASSES), shape=(-1, CLASSES))
        
    with tf.name_scope('total_loss'):
        with tf.name_scope('loss_xy'):
            loss_xy = tf.reduce_sum(tf.square(masked_pred_xy-masked_label_xy))
            tf.summary.scalar('loss_xy', loss_xy)

        with tf.name_scope('loss_wh'):
            loss_wh = tf.reduce_sum(tf.square(tf.sqrt(masked_pred_wh)-tf.sqrt(masked_label_wh)))
            tf.summary.scalar('loss_wh', loss_wh)

        with tf.name_scope('loss_obj'):
            ones = tf.ones(tf.shape(masked_pred_o))
            loss_obj = tf.reduce_sum(tf.square(masked_pred_o - ones))
            tf.summary.scalar('loss_obj', loss_obj)

        with tf.name_scope('loss_no_obj'):
            loss_no_obj = tf.reduce_sum(tf.square(masked_pred_no_o))
            tf.summary.scalar('loss_no_obj', loss_no_obj)

        with tf.name_scope('loss_class'):
            if cross_entropy:
                loss_c = tf.nn.softmax_cross_entropy_with_logits(logits = masked_pred_c, labels = masked_label_c)
                tf.summary.scalar('loss_cross_entropy', loss_c)
            else:
                loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c))
                tf.summary.scalar('loss_squared_error', loss_c)
        
        loss = tf.reduce_mean(COORD*(loss_xy + loss_wh) + loss_obj + NOOBJ * loss_no_obj + loss_c)
        tf.summary.scalar('loss', loss)
        
        return loss

if __name__ == '__main__':
    # Loss Test
    yolo_var = tf.placeholder(tf.float32, [None, 13, 13, 5, 25])
    gt_var = tf.placeholder(tf.float32, [None, 13, 13, 5, 6])

    loss = yolo_loss(yolo_var, gt_var)
    print(loss)
