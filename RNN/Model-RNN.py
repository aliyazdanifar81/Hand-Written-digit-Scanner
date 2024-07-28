#!/usr/bin/env python
# coding: utf-8

# In[169]:


import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split


# # Functions

# **Binarizing**

# In[170]:


def to_bin(images: list):
    for i in range(len(images)):
        gray = cv2.GaussianBlur(images[i], (11,11), 0)
        images[i] = cv2.ximgproc.niBlackThreshold(gray, 255, cv2.THRESH_BINARY_INV, 41, -0.1,
                                            binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)
    return images


# **apply padding on images**

# In[171]:


def pad_images(images):
    padded_images = []
    for img in images:
        padded_img = np.zeros((max_height, max_width))
        h, w = img.shape
        padded_img[max_height//2 - h//2:max_height//2 + (h - h//2), max_width//2 - w//2:max_width//2 + w - (w//2)] = img
        padded_images.append(padded_img)
    return np.array(padded_images)


# **apply padding on labels**

# In[172]:


def pad_labels(labels, max_len: int):
    padded_labels = np.ones((len(labels), max_len), dtype=int) * -1
    for i, label in enumerate(labels):
        label_digits = list(map(int, str(label)))
        padded_labels[i, :len(label_digits)] = label_digits
    return padded_labels


# **loss function**

# In[173]:


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        # At test time, just return the computed predictions
        return y_pred


# **model**

# In[174]:


def build_rnn_model():
    img_height, img_width = 108, 363
    input_img = layers.Input(
            shape=(108, 363, 1), name="image", dtype="float32"
        )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")
    
    # First conv block
    x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            name="Conv1",
        )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    
    # Second conv block
    x = layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            name="Conv2",
        )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    # third conv block
    x = layers.Conv2D(
            512,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            name="Conv3",
        )(x)
    x = layers.AveragePooling2D((2, 2), name="AVGpool1")(x)
    x = layers.Reshape((-1, x.shape[-1] * x.shape[-2]))(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    # Output layer
    x = layers.Dense(11, activation='softmax', name="dense2")(x)
    output = CTCLayer(name="ctc_loss")(labels, x)
    # Define the model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


# # Fetch Dataset

# In[175]:


path = './../Dataset/ORAND-CAR-2014/'


# **Read the CAR-A test and train**<br>Read as gray scale image

# In[176]:


xa_train, xa_test = [cv2.cvtColor(cv2.imread(f'{path}/CAR-A/a_train_images/{img}'), cv2. COLOR_BGR2GRAY) for img in os.listdir(f'{path}/CAR-A/a_train_images')], [cv2.cvtColor(cv2.imread(f'{path}/CAR-A/a_test_images/{img}'), cv2. COLOR_BGR2GRAY) for img in os.listdir(f'{path}/CAR-A/a_test_images')]


# **Read the CAR-B test and train**<br>Read as gray scale image

# In[177]:


xb_train, xb_test = [cv2.cvtColor(cv2.imread(f'{path}/CAR-B/b_train_images/{img}'), cv2. COLOR_BGR2GRAY) for img in os.listdir(f'{path}/CAR-B/b_train_images')], [cv2.cvtColor(cv2.imread(f'{path}/CAR-B/b_test_images/{img}'), cv2. COLOR_BGR2GRAY) for img in os.listdir(f'{path}/CAR-B/b_test_images')]


# **Concatinate**

# In[178]:


x_train, x_test = xa_train + xb_train, xa_test + xb_test


# **Create labels as well**

# In[179]:


ya_train, ya_test, yb_train, yb_test = [], [], [], []
with open(f'{path}CAR-A/a_train_gt.txt', 'r') as a_train, open(f'{path}CAR-A/a_test_gt.txt', 'r') as a_test, open(f'{path}CAR-B/b_train_gt.txt', 'r') as b_train, open(f'{path}CAR-B/b_test_gt.txt', 'r') as b_test:
    for i in a_train:
        ya_train.append(i.split()[1])
    for i in a_test:
        ya_test.append(i.split()[1])
    for i in b_train:
        yb_train.append(i.split()[1])
    for i in b_test:
        yb_test.append(i.split()[1])
y_train, y_test = np.array([int(i) for i in ya_train + yb_train]), np.array([int(i) for i in ya_test + yb_test])


# plot several sample of dataset **before** preprocessing

# In[180]:


fig, ax = plt.subplots(5, 10, figsize=(40, 40))
ax = ax.flatten()
for i in range(50):
    ax[i].imshow(x_train[i], cmap='gray')
    ax[i].title.set_text(y_train[i])


# # Preprocessing

# **Noise Reduction & Binarizing**

# In[181]:


x_train, x_test = to_bin(x_train), to_bin(x_test)


# **find maximum hight and width to padding**

# In[182]:


max_height, max_width = max([sample.shape[0] for sample in xa_train+xa_test+xb_train+xb_test]), max([sample.shape[1] for sample in xa_train+xa_test+xb_train+xb_test])


# In[183]:


print(f'Max height: {max_height}\nMax width: {max_width}')


# **Apply Padding**

# In[184]:


x_train, x_test = pad_images(x_train), pad_images(x_test)


# In[185]:


y_train, y_test = pad_labels(y_train, 8).astype(np.int32), pad_labels(y_test, 8).astype(np.int32)


# In[186]:


y_train.shape


# plot several sample of dataset **after** preprocessing

# In[187]:


fig, ax = plt.subplots(5, 10, figsize=(40, 40))
ax = ax.flatten()
for i in range(50):  
    ax[i].imshow(x_train[i], cmap='gray')
    ax[i].title.set_text(y_train[i])


# **Normalizing**

# In[188]:


x_train, x_test = x_train /255., x_test/255.


# split test to **Test** and **Validation**

# In[189]:


x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.3, random_state=42)


# # Model 

# In[190]:


model = build_rnn_model()


# In[191]:


model.summary()


# In[192]:


model.compile(optimizer='adam', loss=CTCLayer())


# In[193]:


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))


# In[194]:


his = model.fit(train_dataset, validation_data=val_dataset, batch_size=128, epochs=80)


# In[ ]:


plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


img = x_train[8]


# In[ ]:


plt.imshow(img, cmap='gray')


# In[ ]:


predictions = model.predict(img.reshape(-1, 108, 363, 1)/255.)


# In[ ]:


decoded_predictions = tf.keras.backend.ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1])[0][0]


# In[ ]:


predicted_string = ''.join([str(d) for d in decoded_predictions.numpy().flatten() if d != -1])
print(predicted_string)


# In[ ]:


len(predictions[0][0])


# In[ ]:


for i in predictions[0][0]:
    print(i.argmax())


# In[ ]:




