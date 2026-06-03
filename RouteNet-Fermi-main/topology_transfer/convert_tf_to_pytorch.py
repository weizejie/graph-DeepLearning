"""
TF Checkpoint -> PyTorch state_dict Converter
==============================================
Converts RouteNet-Fermi TF checkpoints (fat_tree/ckpt_dir_128/) to
PyTorch state_dict for use with RouteNetFermiPyTorch.

CRITICAL GATE ORDERING FIX (2026-06-01)
========================================
After empirical verification with find_gru_formula.py, we confirmed:

  TF GRUCell kernel layout (columns):    [z=update, r=reset, n=new]
  PyTorch GRUCell weight_ih layout (rows): [r,         z,      n   ]

  TF recurrent_kernel layout (columns):  [z=update, r=reset, n=new]
  PyTorch weight_hh layout (rows):        [r,         z,      n   ]

  TF bias layout (rows 0,1):            [z_ih, z_hh], [r_ih, r_hh], [n_ih, n_hh]
  PyTorch bias_ih / bias_hh (flat):      [r_ih], [z_ih], [n_ih]  /  [r_hh], [z_hh], [n_hh]

So TF weights must be REORDERED before loading into PyTorch:
  PT rows [reset]   <- TF columns [reset_col=1]
  PT rows [update]  <- TF columns [update_col=0]
  PT rows [new]     <- TF columns [new_col=2]
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import tensorflow as tf
import numpy as np


def dense_weight(x):
    return x.T


def dense_bias(x):
    return x


# TF gate indices
TF_UPDATE = 0   # TF kernel column 0:H
TF_RESET  = 1   # TF kernel column H:2H
TF_NEW    = 2   # TF kernel column 2H:3H

# PyTorch gate row indices
PT_RESET  = 0   # PT weight row 0:H
PT_UPDATE = 1   # PT weight row H:2H
PT_NEW    = 2   # PT weight row 2H:3H


def reorder_gru_weights(tf_kernel, tf_rec_kernel, tf_bias):
    """
    Reorder TF GRU weights to PyTorch GRU layout.

    TF gates (columns): [z=update, r=reset, n=new]
    PT gates (rows):    [r,          z,        n   ]

    Returns: (pt_weight_ih, pt_weight_hh, pt_bias_ih, pt_bias_hh)
    """
    H = tf_bias.shape[1] // 3  # hidden size

    # --- Reorder kernel columns: TF(col) -> PT(row) ---
    # TF kernel: (input, 3H), PT weight_ih: (3H, input)
    # TF columns [z,r,n] -> PT rows [r,z,n]
    reordered_kernel = np.concatenate([
        tf_kernel[:, TF_RESET  * H : (TF_RESET  + 1) * H],   # TF reset col -> PT row 0:H
        tf_kernel[:, TF_UPDATE * H : (TF_UPDATE + 1) * H],   # TF update col -> PT row H:2H
        tf_kernel[:, TF_NEW    * H : (TF_NEW    + 1) * H],   # TF new col -> PT row 2H:3H
    ], axis=1)  # (input, 3H) = (I, 96) -> PyTorch (3H, I) = (96, I) via .T

    # --- Reorder recurrent kernel columns: TF(col) -> PT(row) ---
    # TF rec_kernel: (hidden, 3H), PT weight_hh: (3H, hidden)
    reordered_rec_kernel = np.concatenate([
        tf_rec_kernel[:, TF_RESET  * H : (TF_RESET  + 1) * H],
        tf_rec_kernel[:, TF_UPDATE * H : (TF_UPDATE + 1) * H],
        tf_rec_kernel[:, TF_NEW    * H : (TF_NEW    + 1) * H],
    ], axis=1)  # (hidden, 3H)

    # --- Reorder bias rows: TF(row) -> PT(flat) ---
    # TF bias: (2, 3H) -> row 0: [z_ih, r_ih, n_ih], row 1: [z_hh, r_hh, n_hh]
    # PyTorch: bias_ih [r_ih, z_ih, n_ih], bias_hh [r_hh, z_hh, n_hh]
    tf_bias_ih = tf_bias[0]   # (3H,)
    tf_bias_hh = tf_bias[1]   # (3H,)

    reordered_bias_ih = np.concatenate([
        tf_bias_ih[TF_RESET  * H : (TF_RESET  + 1) * H],
        tf_bias_ih[TF_UPDATE * H : (TF_UPDATE + 1) * H],
        tf_bias_ih[TF_NEW    * H : (TF_NEW    + 1) * H],
    ])  # (3H,)

    reordered_bias_hh = np.concatenate([
        tf_bias_hh[TF_RESET  * H : (TF_RESET  + 1) * H],
        tf_bias_hh[TF_UPDATE * H : (TF_UPDATE + 1) * H],
        tf_bias_hh[TF_NEW    * H : (TF_NEW    + 1) * H],
    ])  # (3H,)

    return reordered_kernel, reordered_rec_kernel, reordered_bias_ih, reordered_bias_hh


# Mapping: (TF var name, PT key, transform)
MAPPING = [
    # -- Path Embedding (input=17, hidden=32) ---------------------------------
    ("path_embedding/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "path_embedding.net.0.weight", dense_weight),
    ("path_embedding/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "path_embedding.net.0.bias",   dense_bias),
    ("path_embedding/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "path_embedding.net.2.weight", dense_weight),
    ("path_embedding/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "path_embedding.net.2.bias",   dense_bias),

    # -- Queue Embedding (input=5, hidden=32) ---------------------------------
    ("queue_embedding/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "queue_embedding.net.0.weight", dense_weight),
    ("queue_embedding/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "queue_embedding.net.0.bias",   dense_bias),
    ("queue_embedding/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "queue_embedding.net.2.weight", dense_weight),
    ("queue_embedding/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "queue_embedding.net.2.bias",   dense_bias),

    # -- Link Embedding (input=5, hidden=32) ----------------------------------
    ("link_embedding/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "link_embedding.net.0.weight", dense_weight),
    ("link_embedding/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "link_embedding.net.0.bias",   dense_bias),
    ("link_embedding/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "link_embedding.net.2.weight", dense_weight),
    ("link_embedding/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "link_embedding.net.2.bias",   dense_bias),

    # -- Path GRU (input=Q+L=64, hidden=32) -----------------------------------
    # TF: kernel(64,96), rec_kernel(32,96), bias(2,96) -> gate order [z,r,n]
    # PT: weight_ih(96,64), weight_hh(96,32), bias_ih(96,), bias_hh(96,) -> gate order [r,z,n]
    ("path_update/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "path_gru.weight_ih", None),   # special: handled by reorder_gru_weights
    ("path_update/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "path_gru.weight_hh", None),
    ("path_update/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "path_gru.bias_ih", None),      # bias_ih
    ("path_update/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "path_gru.bias_hh", None),     # bias_hh

    # -- Queue GRU (input=P=32, hidden=32) -------------------------------------
    ("queue_update/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "queue_gru.weight_ih", None),
    ("queue_update/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "queue_gru.weight_hh", None),
    ("queue_update/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "queue_gru.bias_ih", None),
    ("queue_update/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "queue_gru.bias_hh", None),

    # -- Link GRU (input=Q=32, hidden=32) -------------------------------------
    ("link_update/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "link_gru.weight_ih", None),
    ("link_update/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "link_gru.weight_hh", None),
    ("link_update/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "link_gru.bias_ih", None),
    ("link_update/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "link_gru.bias_hh", None),

    # -- Path Readout (input=32, hidden=16, output=1) --------------------------
    ("readout_path/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "readout_path.net.0.weight", dense_weight),
    ("readout_path/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "readout_path.net.0.bias",   dense_bias),
    ("readout_path/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "readout_path.net.2.weight", dense_weight),
    ("readout_path/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "readout_path.net.2.bias",   dense_bias),
    ("readout_path/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE",
     "readout_path.net.4.weight", dense_weight),
    ("readout_path/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE",
     "readout_path.net.4.bias",   dense_bias),
]


# Group entries that need reordering (bias entry appears twice for each GRU)
GRU_REORDER_PAIRS = {
    'path_gru':  ('path_update/kernel/.ATTRIBUTES/VARIABLE_VALUE',
                  'path_update/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE',
                  'path_update/bias/.ATTRIBUTES/VARIABLE_VALUE'),
    'queue_gru': ('queue_update/kernel/.ATTRIBUTES/VARIABLE_VALUE',
                  'queue_update/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE',
                  'queue_update/bias/.ATTRIBUTES/VARIABLE_VALUE'),
    'link_gru':  ('link_update/kernel/.ATTRIBUTES/VARIABLE_VALUE',
                  'link_update/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE',
                  'link_update/bias/.ATTRIBUTES/VARIABLE_VALUE'),
}


def convert(tf_ckpt_path, output_path):
    print(f"Loading TF checkpoint: {tf_ckpt_path}")
    reader = tf.train.load_checkpoint(tf_ckpt_path)
    shapes = reader.get_variable_to_shape_map()

    converted = {}

    # --- Handle Dense layers (simple transforms) ---
    for tf_name, pt_key, transform in MAPPING:
        if transform is not None:
            if tf_name not in shapes:
                print(f"  [SKIP] {tf_name} not found")
                continue
            tf_tensor = reader.get_tensor(tf_name)
            result = transform(tf_tensor)
            converted[pt_key] = torch.from_numpy(result.astype(np.float32))

    # --- Handle GRU layers (reorder gate dimensions) ---
    for gru_name, (kernel_key, rec_key, bias_key) in GRU_REORDER_PAIRS.items():
        if kernel_key not in shapes:
            print(f"  [SKIP] GRU {gru_name}: kernel not found")
            continue
        K = reader.get_tensor(kernel_key)
        R = reader.get_tensor(rec_key)
        B = reader.get_tensor(bias_key)
        print(f"  GRU {gru_name}: K={K.shape}, R={R.shape}, B={B.shape}")

        w_ih, w_hh, b_ih, b_hh = reorder_gru_weights(K, R, B)
        converted[f"{gru_name}.weight_ih"] = torch.from_numpy(w_ih.astype(np.float32).T)
        converted[f"{gru_name}.weight_hh"] = torch.from_numpy(w_hh.astype(np.float32).T)
        converted[f"{gru_name}.bias_ih"] = torch.from_numpy(b_ih.astype(np.float32))
        converted[f"{gru_name}.bias_hh"] = torch.from_numpy(b_hh.astype(np.float32))

    # --- Validate shapes ---
    from topology_transfer.routenet_fermi_pytorch import RouteNetFermiPyTorch
    pt_model = RouteNetFermiPyTorch()
    pt_state = pt_model.state_dict()

    print("\nShape comparison (converted vs PyTorch expected):")
    all_ok = True
    for pt_key in sorted(converted.keys()):
        tf_val = converted[pt_key]
        if pt_key in pt_state:
            pt_val = pt_state[pt_key]
            match = "OK" if tf_val.shape == pt_val.shape else f"MISMATCH (PT wants {tuple(pt_val.shape)})"
            if match != "OK":
                all_ok = False
            print(f"  {pt_key}: converted={tuple(tf_val.shape)}  PT={tuple(pt_val.shape)}  [{match}]")
        else:
            print(f"  {pt_key}: converted={tuple(tf_val.shape)}  PT=[NOT FOUND]")
            all_ok = False

    if not all_ok:
        print("\n[ERROR] Shape mismatches detected.")
        return None

    print("\nAll shapes match! Saving checkpoint.")

    ckpt = {
        'model_state_dict': converted,
        'epoch': 2,
        'val_mape': 0.59,
        'source': tf_ckpt_path,
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"Saved to: {output_path}")
    return converted


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_ckpt', type=str,
                        default='../fat_tree/ckpt_dir_128/02-0.59')
    parser.add_argument('--output', type=str,
                        default='./checkpoints/converted_from_tf.pt')
    args = parser.parse_args()
    convert(args.tf_ckpt, args.output)
