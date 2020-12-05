import cv2
import numpy as np
import tensorflow as tf
from model import generator, discriminator

path = "test_image/input/0234.png"

X = cv2.imread(path)
X = cv2.resize(X,(24,24), interpolation = cv2.INTER_AREA)
X = np.reshape(X, (1,24,24,3))
X_batch = tf.cast(X, tf.float32)

model = tf.keras.models.Sequential()
model.add(generator())
model.add(discriminator())

generator, discriminator = model.layers
generator.load_weights("weights/gan_generator.h5")
Y = generator(X_batch)
cv2.imshow("LR",X[0])
cv2.imshow("HR", Y[0].numpy())
name = path.split(".")
name = name[0].split("/")
cv2.imwrite("test_image/output/{}.png".format(str(name[-1])), Y[0].numpy())