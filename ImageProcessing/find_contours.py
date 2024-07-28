import cv2
import matplotlib.pyplot as plt

from preprocess import padding


def con_component(img):
    # plt.imshow(img, cmap='gray')
    # plt.show()
    number_images = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=4)
    for i in range(1, num_labels):  # Skip the background label 0
        x, y, w, h, area = stats[i]
        plt.imshow(img[y:y + h, x:x + w], cmap='gray')
        plt.show()
        number_images.append(img[y:y + h, x:x + w])
    return number_images


def find_cont(img):
    # plt.imshow(img, cmap='gray')
    # plt.show()
    number_images = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        number_image = img[y:y + h, x:x + w]
        number_images.append((padding(number_image), (x, y, w, h)))
        # plt.imshow(padding(number_image), cmap='gray')
        # plt.show()
    res = sorted(number_images, key=lambda item: item[1][0])
    return res
