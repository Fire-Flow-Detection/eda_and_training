# Metrics copied from the assessment matrix section of the notebook, which
# Nova authored. Thanks, Nova!

import tensorflow as tf

def _binarize_masks(y_true, y_pred, threshold=0.5):

    # Ensure ground-truth is binary (handle cases where it's not exactly 0/1)
    y_true_bin = tf.cast(y_true > 0.5, tf.float32)

    # Convert predicted probabilities to binary mask
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)

    return y_true_bin, y_pred_bin

# IoU
def IoU_metric(y_true, y_pred, threshold=0.5):

    y_true_bin, y_pred_bin = _binarize_masks(y_true, y_pred, threshold)

    # Intersection: pixels that are 1 in both prediction and ground truth
    intersection = tf.reduce_sum(y_true_bin * y_pred_bin)

    # Union: total number of pixels that are 1 in either prediction or ground truth
    union = tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin) - intersection

    # Avoid division by zero using divide_no_nan
    return tf.math.divide_no_nan(intersection, union)

# F1 for segmentation
def dice_metric(y_true, y_pred, threshold=0.5):

    y_true_bin, y_pred_bin = _binarize_masks(y_true, y_pred, threshold)

    intersection = tf.reduce_sum(y_true_bin * y_pred_bin)
    sum_ = tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin)

    # Dice = 2 * intersection / (|A| + |B|)
    return tf.math.divide_no_nan(2.0 * intersection, sum_)

# Precision = TP / (TP + FP)
def precision_metric(y_true, y_pred, threshold=0.5):

    y_true_bin, y_pred_bin = _binarize_masks(y_true, y_pred, threshold)

    # True Positive: predicted = 1 and true = 1
    tp = tf.reduce_sum(y_true_bin * y_pred_bin)

    # Predicted Positive: predicted = 1
    predicted_pos = tf.reduce_sum(y_pred_bin)

    return tf.math.divide_no_nan(tp, predicted_pos)

# Recall = TP / (TP + FN)
def recall_metric(y_true, y_pred, threshold=0.5):

    y_true_bin, y_pred_bin = _binarize_masks(y_true, y_pred, threshold)

    # True Positive: predicted = 1 and true = 1
    tp = tf.reduce_sum(y_true_bin * y_pred_bin)

    # Actual Positive: true = 1
    actual_pos = tf.reduce_sum(y_true_bin)

    return tf.math.divide_no_nan(tp, actual_pos)