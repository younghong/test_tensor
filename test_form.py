import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

img_height = 180
img_width = 180

def load():
    return load_model('model_form.h5')

model = load()

#flower_name = 'band_sample.png'
#flower_name = 'freesample.png'
flower_name = 'post_test1.jpg'
#flower_name = 'bd_UBImage.png'
#flower_name = './form_types/free/1.jpg'

img = cv.imread(flower_name, cv.IMREAD_COLOR)
img = cv.resize(img, dsize=(img_width, img_height))
cv.imshow('result2', img)

cv.waitKey(0)


img = (np.expand_dims(img,0))
print(img.shape)

cv.waitKey(0)

predictions_single = model.predict(img)
print(predictions_single)

cv.waitKey(0)


sel = np.argmax(predictions_single[0])
print(sel)

cv.waitKey(0)

class_names = ['band', 'free','label']
print(class_names[sel])
