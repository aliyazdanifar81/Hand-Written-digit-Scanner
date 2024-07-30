import os
import sys

sys.path.append('./../')
from Metrics import cer
import numpy as np
from preprocess import preprocess as pep, resizing
from find_contours import find_cont, con_component
from importData import fetch_data
import tensorflow as tf

if __name__ == '__main__':
    predicted = []
    x, y = fetch_data()
    for sample in x:
        preprocessed_img = pep(sample).astype('uint8')
        numbers = find_cont(preprocessed_img)
        final_images = resizing(numbers)
        model = tf.keras.models.load_model('./my_model.h5')
        detected = 0
        for num in final_images:
            num = num / 255.0
            pre = model.predict(num.reshape(-1, 28, 28, 1))
            if len(num[num == 1]) / len(num[num == 0]) > 42:
                detected += np.argmax(pre)
                detected *= 10
        predicted.append(detected)
res, same = 0, 0
for i in range(len(y)):
    if y[i] == predicted[i]:
        same += 1
    res += cer(y[i], predicted[i])
print(f'Mean of error rate is {res / len(y)}')
