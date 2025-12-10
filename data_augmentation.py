import numpy as np
import tensorflow as tf
from typing import Literal

def random_geometric_augment(x, y):
    # flip horizontally
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    # flip vertically
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
    # random 0, 90, 180, 270 rotate
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    y = tf.image.rot90(y, k)
    return x, y

def filter_nofire_fraction(x, y, keep_ratio=0.3):
    is_nofire = tf.equal(tf.reduce_sum(y), 0)

    rnd = tf.random.uniform(())
    keep = tf.logical_or(tf.logical_not(is_nofire), rnd < keep_ratio)
    return keep

def get_fire_bbox(same_day_fire, next_day_fire):
    fire_coords = np.where((same_day_fire) > 0 | (next_day_fire) > 0)
    if len(fire_coords[0]) == 0:
        return None
    return (fire_coords[0].min(), fire_coords[0].max(),
            fire_coords[1].min(), fire_coords[1].max())

# TODO
def smart_crop(x, y, target_height, target_width,
               neg_crop_mode: Literal["center", "random"]):
    # x fire
    same_day_fire = x[:, :, -1].copy()
    # y fire
    next_day_fire = y[:, :, -1].copy()
    row_min, row_max, col_min, col_max = get_fire_bbox(same_day_fire, next_day_fire)
    # check height and width
    pass