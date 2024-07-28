import os

import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess as pep, resizing
from find_contours import find_cont, con_component
import tensorflow as tf

if __name__ == '__main__':
    for sample in os.listdir('./../Dataset/ORAND-CAR-2014/CAR-A/a_train_images'):
        data_path = f'./../Dataset/ORAND-CAR-2014/CAR-A/a_train_images/{sample}'
        preprocessed_img = pep(data_path).astype('uint8')
        plt.imshow(preprocessed_img, cmap='gray')
        plt.show()
        numbers = find_cont(preprocessed_img)
        final_images = resizing(numbers)
        model = tf.keras.models.load_model('./my_model2.h5')
        detected = 0
        for num in final_images:
            num = num / 255.0
            plt.imshow(num, cmap='gray')
            plt.show()
            pre = model.predict(num.reshape(-1, 28, 28, 1))
            print(f'Maybe : {np.argmax(pre)}')
            if np.max(pre) > 0.8:
                detected += np.argmax(pre)
                detected *= 10
        print(detected // 10)
