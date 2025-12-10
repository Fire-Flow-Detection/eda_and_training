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
    fire_coords = np.where((same_day_fire > 0) | (next_day_fire > 0))
    if len(fire_coords[0]) == 0:
        return None, None, None, None
    return (fire_coords[0].min(), fire_coords[0].max(),
            fire_coords[1].min(), fire_coords[1].max())


def smart_crop(x, y, target_height, target_width,
               neg_crop_mode: Literal["center", "random"] = "center",
               min_context_margin: float = 0.5):
    """
    Smart crop around fire regions with context, or random/center crop if no fire.
    """
    h, w = x.shape[:2]

    if h < target_height or w < target_width:
        raise ValueError(f"Image ({h}x{w}) must be at least target size "
                         f"({target_height}x{target_width})")

    same_day_fire = x[:, :, -1]
    next_day_fire = y[:, :, 0]

    row_min, row_max, col_min, col_max = get_fire_bbox(same_day_fire, next_day_fire)

    if row_min is None:
        if neg_crop_mode == "center":
            start_row = (h - target_height) // 2
            start_col = (w - target_width) // 2
        else:
            start_row = np.random.randint(0, h - target_height + 1)
            start_col = np.random.randint(0, w - target_width + 1)

        end_row = start_row + target_height
        end_col = start_col + target_width
        cropped_x = x[start_row:end_row, start_col:end_col, :]
        cropped_y = y[start_row:end_row, start_col:end_col, :]

        return cropped_x, cropped_y

    else:
        bbox_height = row_max - row_min
        bbox_width = col_max - col_min
        context_padding = int(max(bbox_height, bbox_width) * min_context_margin)

        center_row = (row_min + row_max) // 2
        center_col = (col_min + col_max) // 2

        if ((bbox_height + 2 * context_padding <= target_height) and
                (bbox_width + 2 * context_padding <= target_width)):

            start_row = center_row - target_height // 2
            start_col = center_col - target_width // 2

            if start_row < 0:
                start_row = 0
            elif start_row + target_height > h:
                start_row = h - target_height

            if start_col < 0:
                start_col = 0
            elif start_col + target_width > w:
                start_col = w - target_width

            end_row = start_row + target_height
            end_col = start_col + target_width
            cropped_x = x[start_row:end_row, start_col:end_col, :]
            cropped_y = y[start_row:end_row, start_col:end_col, :]

            return cropped_x, cropped_y

        else:
            target_aspect = target_width / target_height
            ideal_height = bbox_height + 2 * context_padding
            ideal_width = bbox_width + 2 * context_padding

            if ideal_width / ideal_height < target_aspect:
                ideal_width = int(ideal_height * target_aspect)
            else:
                ideal_height = int(ideal_width / target_aspect)

            start_row = center_row - ideal_height // 2
            start_col = center_col - ideal_width // 2

            if start_row < 0:
                start_row = 0
            elif start_row + ideal_height > h:
                start_row = h - ideal_height

            if start_col < 0:
                start_col = 0
            elif start_col + ideal_width > w:
                start_col = w - ideal_width

            end_row = start_row + ideal_height
            end_col = start_col + ideal_width
            large_crop_x = x[start_row:end_row, start_col:end_col, :]
            large_crop_y = y[start_row:end_row, start_col:end_col, :]

            cropped_x = tf.image.resize(
                large_crop_x,
                (target_height, target_width),
                method='bilinear'
            ).numpy()

            cropped_y = tf.image.resize(
                large_crop_y,
                (target_height, target_width),
                method='nearest'
            ).numpy()

            cropped_y = (cropped_y > 0.5).astype("float32")

            return cropped_x, cropped_y