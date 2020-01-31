import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import pandas as pd
import tensorflow as tf
import preprocess
from preprocess import rescale_segment as rescale_segment
from preprocess import extract_segments as extract_segments

#original image
img = cv2.imread('./images/EGg1WSp.png', 0)
img = cv2.resize(img, (368, 75))
plt.imshow(img, cmap='gray')
#plt.show()

#eroded image
temp = img[0: 350, 0: 500]
kernel = np.ones([3, 3])
temp = cv2.erode(temp, kernel, iterations=1)
plt.imshow(temp, cmap='gray')
#plt.show()

#segmentation
segments = extract_segments(temp, 30, reshape=1, size=[28, 28],
                            threshold=40, area=10, ker=1)

plt.figure(figsize=[15, 15])
plt.subplot(1, 16, 1)
plt.imshow(temp, cmap='gray')

# print(segments)
for j in range(len(segments)):
    plt.subplot(1, 16, 2+j)
    plt.imshow(segments[j], cmap='gray')
plt.show()

class_labels = {str(x): x for x in range(10)}
class_labels.update({'+': 10, 'times': 11, '-': 12})
label_class = dict(zip(class_labels.values(), class_labels.keys()))

 # load model
cnn_ver1 = preprocess.load_models()

cnn_pred = ''
for i in range(len(segments)):
     # plot each segment
     plt.subplot(1, 16, 1 + i)
     temp = segments[i]
     plt.imshow(temp, cmap='gray')
     temp = temp.reshape(1, -1)
     pred = preprocess.predict(temp, label_class, 0, cnn_ver1)
     cnn_pred += pred
     plt.show()

print('CNN_ver2 model result : ', cnn_pred)
