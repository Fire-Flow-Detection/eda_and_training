import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import rasterio
import tensorflow as tf
from typing import Literal, Optional
from google.cloud import storage
from configs.data_config import EXPECTED_BANDS, EXPECTED_WIDTH, EXPECTED_HEIGHT

# def get_year_from_blob(blob_name):
#     """
#     Gets year from a blob name formatted thusly:
#     year/fire_id/YYYY_MM_DD.tif
#     """
#     parts = blob_name.split("/")
#     year = parts[0]
#     return year

def collect_paths(source_dir:str,
                  mode: Literal["train", "val", "test"],
                  client:Optional[storage.Client] = None):
    source_dir = f"{source_dir}/{mode}"
    if source_dir.lower().startswith("gs://"):
        if not client:
            client = storage.Client()
        bucket_name, prefix = (source_dir.replace("gs://", "")
                               .split("/", 1))
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        paths = [f"gs://{bucket_name}/{blob.name}"
                 for blob in blobs
                 if blob.name.lower().endswith(('.tif', '.tiff'))]
    else:
        paths = []
        for root, _, files in os.walk(source_dir):
            for filename in files:
                if filename.lower().endswith(('.tif', '.tiff')):
                    full_path = os.path.join(root, filename)
                    paths.append(full_path)
    print(f"{mode}: found {len(paths)} tif files")
    return sorted(paths)

def validate_dimensions(sample, height, width, bands):
    if height < EXPECTED_HEIGHT:
        rows_to_add = EXPECTED_HEIGHT - height
        bottom_row = sample[:, -1:, :]
        sample = np.concatenate(
            [sample] + [bottom_row] * rows_to_add, axis=1)
    if width < EXPECTED_WIDTH:
        cols_to_add = EXPECTED_WIDTH - width
        rightmost_col = sample[:, :, -1:]
        sample = np.concatenate(
            [sample] + [rightmost_col] * cols_to_add, axis=2)
    if len(bands) != EXPECTED_BANDS:
        raise ValueError(f"Image has {len(bands)} bands, expected 19.")
    if width > EXPECTED_WIDTH or height > EXPECTED_HEIGHT:
        raise ValueError(f"Image is too large. Max size is 172x172 px")
    return sample

def img_from_path(path:str):
    if hasattr(path, 'numpy'):  # It's a TensorFlow tensor
        path = path.numpy()
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    with rasterio.open(path) as src:
        item = src.read().astype("float32")
        item[~np.isfinite(item)] = 0.0
        item_bands = [src.descriptions[i] for i in range(src.count)]
        item_height = src.height
        item_width = src.width
        item = validate_dimensions(item, item_height, item_width,
                                       bands=item_bands)
        label_idx = None
        for i, name in enumerate(item_bands):
            if name and str(name).lower().strip() == "label":
                label_idx = i
                break
        if label_idx is None:
            raise ValueError(f"No 'label' band found in {path}. Bands: {item_bands}")
        features = np.delete(item, label_idx, axis=0)
        label = item[label_idx, :, :]
        x = features.transpose(1, 2, 0).astype("float32")
        y = np.expand_dims(label, axis=-1).astype("float32")
        y = (y > 0).astype("float32")
        return x, y

def load_tif_tf(path):
    """
    Wrapper for tf.data: path (string tensor) → (x, y)
    """
    x, y = tf.py_function(img_from_path, [path], [tf.float32, tf.float32])
    x.set_shape([None, None, None])
    y.set_shape([None, None, 1])
    return x, y

def make_dataset(source_dir, mode: Literal["train", "val", "test"],
                 batching = True, batch_size=8, shuffle=True):
    """
    Create tf.data.Dataset for train / val / test.
    """
    paths = collect_paths(source_dir, mode, client=None)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(len(paths))
    ds = ds.map(load_tif_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if batching:
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds