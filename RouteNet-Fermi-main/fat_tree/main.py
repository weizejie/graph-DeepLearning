import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from data_generator import input_fn

import sys

sys.path.append('../')
from delay_model import RouteNet_Fermi

N = 128

# 请修改为你的数据集路径 (Windows 路径示例)
# 数据已放在项目目录下的 fat128 文件夹中
DATA_BASE_DIR = r'D:\newdownload\RouteNet-Fermi-main (1)\RouteNet-Fermi-main\fat128'

TRAIN_PATH = os.path.join(DATA_BASE_DIR, 'train')
VALIDATION_PATH = os.path.join(DATA_BASE_DIR, 'test')
TEST_PATH = os.path.join(DATA_BASE_DIR, 'test')

# 检查路径是否存在
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"数据集路径不存在: {TRAIN_PATH}\n请修改 main.py 中的 DATA_BASE_DIR 变量")

print(f"训练数据路径: {TRAIN_PATH}")
print(f"验证数据路径: {VALIDATION_PATH}")

ds_train = input_fn(TRAIN_PATH, shuffle=True)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()

ds_validation = input_fn(VALIDATION_PATH, shuffle=False)
ds_validation = ds_validation.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

model = RouteNet_Fermi()

loss_object = tf.keras.losses.MeanAbsolutePercentageError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False)

ckpt_dir = f'./ckpt_dir_{N}'
latest = tf.train.latest_checkpoint(ckpt_dir)

if latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode="min",
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=True,
    save_freq='epoch')

model.fit(ds_train,
          epochs=5,
          steps_per_epoch=200,
          validation_data=ds_validation,
          callbacks=[cp_callback],
          use_multiprocessing=True)

ds_test = input_fn(TEST_PATH, shuffle=False)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model.evaluate(ds_test)
