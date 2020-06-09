from tensorflow import keras
model = keras.models.load_model('/Users/a./Desktop/course/model svae')

import cv2
import numpy as np
from keras.preprocessing import image

img = image.load_img('/Users/a./Desktop/IMG_0270.jpg', target_size=(300, 300))
x = image.img_to_array(img)
x = x/255
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)

if classes[0][0]==0:
    print("ITS A HORSE")

else:
    print("ITS A HUMAN")
