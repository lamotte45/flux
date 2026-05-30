import tensorflow as tf

# ============================================================
# TensorBoard Logging Setup
# ============================================================

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="/home/user/barber_ai/logs",
    histogram_freq=1
)

# ============================================================
# Example Model (Replace with your own)
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(128,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Dummy example dataset (replace with your real train_ds / val_ds)
import numpy as np
train_ds = (np.random.rand(1000, 128), np.random.randint(0, 10, 1000))
val_ds = (np.random.rand(200, 128), np.random.randint(0, 10, 200))

# ============================================================
# Train with TensorBoard Logging
# ============================================================

model.fit(
    train_ds[0],
    train_ds[1],
    validation_data=val_ds,
    epochs=10,
    callbacks=[tensorboard_callback]
)

print("Training complete. TensorBoard logs written to /home/user/barber_ai/logs")
