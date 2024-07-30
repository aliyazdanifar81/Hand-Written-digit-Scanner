import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split


def prepare_labels(labels):
    # Assuming labels are a list of strings representing digits
    prepared_labels = []
    for label in labels:
        label_digits = [int(char) for char in str(label)]
        prepared_labels.append(label_digits)
    return tf.ragged.constant(prepared_labels, dtype=tf.int64).to_tensor(default_value=-1)


def res_images(images, target_height=108, target_width=363):
    resized_images = []
    for img in images:
        # Ensure the image is a 3D array (height, width, channels)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)  # Add a channel dimension for grayscale images

        # Resize the image
        resized_img = tf.image.resize(img, [target_height, target_width], method='nearest')
        resized_img = resized_img.numpy()  # Convert back to a numpy array

        resized_images.append(resized_img)

    return np.array(resized_images)


def fetch_data():
    path = './../Dataset/ORAND-CAR-2014/'
    xa_train, xa_test = [cv2.cvtColor(cv2.imread(f'{path}/CAR-A/a_train_images/{img}'), cv2.COLOR_BGR2GRAY) for img in
                         os.listdir(f'{path}/CAR-A/a_train_images')], [
                            cv2.cvtColor(cv2.imread(f'{path}/CAR-A/a_test_images/{img}'), cv2.COLOR_BGR2GRAY) for img in
                            os.listdir(f'{path}/CAR-A/a_test_images')]
    xb_train, xb_test = [cv2.cvtColor(cv2.imread(f'{path}/CAR-B/b_train_images/{img}'), cv2.COLOR_BGR2GRAY) for img in
                         os.listdir(f'{path}/CAR-B/b_train_images')], [
                            cv2.cvtColor(cv2.imread(f'{path}/CAR-B/b_test_images/{img}'), cv2.COLOR_BGR2GRAY) for img in
                            os.listdir(f'{path}/CAR-B/b_test_images')]
    x = xa_train + xb_train + xa_test + xb_test
    ya_train, ya_test, yb_train, yb_test = [], [], [], []
    with open(f'{path}CAR-A/a_train_gt.txt', 'r') as a_train, open(f'{path}CAR-A/a_test_gt.txt', 'r') as a_test, open(
            f'{path}CAR-B/b_train_gt.txt', 'r') as b_train, open(f'{path}CAR-B/b_test_gt.txt', 'r') as b_test:
        for i in a_train:
            ya_train.append(i.split()[1])
        for i in a_test:
            ya_test.append(i.split()[1])
        for i in b_train:
            yb_train.append(i.split()[1])
        for i in b_test:
            yb_test.append(i.split()[1])
    y = ya_train + yb_train + ya_test + yb_test
    return x, y
