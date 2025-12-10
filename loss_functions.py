import tensorflow as tf

def binary_focal_loss(gamma=2.0, alpha=0.75):

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -(y_true * tf.math.log(y_pred) +
               (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        weight = (
            alpha * y_true * tf.pow(1.0 - y_pred, gamma) +
            (1.0 - alpha) * (1.0 - y_true) * tf.pow(y_pred, gamma)
        )

        return tf.reduce_mean(weight * ce)

    return loss_fn

def weighted_dice_loss(y_true, y_pred, pos_weight = 50.0):
    # HEAVILY weighted towards positive class
    smooth = 1e-6
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    weighted_intersection = tf.reduce_sum(y_true_f * y_pred_f * pos_weight)
    weighted_union = tf.reduce_sum(y_true_f * pos_weight) + tf.reduce_sum(y_pred_f)

    return 1.0 - (2.0 * weighted_intersection + smooth) / (weighted_union + smooth)

def dice_loss(y_true, y_pred):

    smooth = 1e-6
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (denom + smooth)


def focal_dice_loss(y_true, y_pred):
    focal = binary_focal_loss(gamma=4.0, alpha=0.9)(y_true, y_pred)
    d = weighted_dice_loss(y_true, y_pred)
    return focal + d