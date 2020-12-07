import cv2
import numpy as np
import tensorflow as tf
from model import generator, discriminator

path = "test_image/input/050.png"

X = cv2.imread(path)
len = X.shape[0]
width = X.shape[1]

print(len)
print(width)

if len > width:
    X = cv2.resize(X,(width,width), interpolation= cv2.INTER_CUBIC)

else:
    X = cv2.resize(X, (len,len),interpolation= cv2.INTER_CUBIC)

size = X.shape[0]
output = np.zeros((size*4,size*4,3))

n_of_section = 1 + int(size/24)

model = tf.keras.models.Sequential()
model.add(generator())
model.add(discriminator())

generator, discriminator = model.layers
generator.load_weights("weights/gan_generator.h5")

X = X/255

for i in range (n_of_section):
    for j in range (n_of_section):

        if (j+1)*24 < size and (i+1)*24 < size:
            input = X[i*24:(i+1)*24,j*24:(j+1)*24,:]
            input = np.reshape(input, (1, 24, 24, 3))
            X_batch = tf.cast(input, tf.float32)
            Y = generator(X_batch)
            output[i*96:(i+1)*96,j*96:(j+1)*96,:] = (Y[0].numpy() + 1)*127.5

        elif (i+1)*24 > size and (j+1)*24 < size:
            input = X[size - 24:size, j*24:(j+1)*24, :]
            input = np.reshape(input, (1, 24, 24, 3))
            X_batch = tf.cast(input, tf.float32)
            Y = generator(X_batch)
            output[size * 4 - 96:size * 4, j*96:(j+1)*96, :] = (Y[0].numpy() + 1)*127.5

        elif (i + 1) * 24 < size and (j + 1) * 24 > size:
            input = X[ i * 24:(i + 1) * 24,size - 24:size, :]
            input = np.reshape(input, (1, 24, 24, 3))
            X_batch = tf.cast(input, tf.float32)
            Y = generator(X_batch)
            output[ i * 96:(i + 1) * 96,size * 4 - 96:size * 4, :] = (Y[0].numpy() + 1)*127.5

        else:
            input = X[size-24:size, size-24:size, :]
            input = np.reshape(input, (1, 24, 24, 3))
            X_batch = tf.cast(input, tf.float32)
            Y = generator(X_batch)
            output[size*4-96:size*4, size*4-96:size*4, :] = (Y[0].numpy() + 1)*127.5

        print("{}/{}".format(i*n_of_section + j+1, n_of_section*n_of_section))

cv2.waitKey(0)
cv2.destroyAllWindows()
name = path.split(".")
name = name[0].split("/")
cv2.imwrite("test_image/output/{}.png".format(str(name[-1])), output)
