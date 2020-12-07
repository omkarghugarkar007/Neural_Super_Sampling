import model
from model import generator, discriminator

from tqdm import tqdm
import cv2

batch_size = 32

def train_dcgan(model,epochs=5):
   

    print("done")
    generator, discriminator = model.layers
    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
    discriminator.trainable = False
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")
    path = '/content/drive/MyDrive/cars_train/'
    for epoch in tqdm(range(epochs)):
      print("Epoch {}/{}".format(epoch + 1, epochs))
      for root, dirnames, filenames in os.walk(path):
        i = 0
        j = 0
        x_train_x = np.zeros((32,24,24,3))
        x_train_y = np.zeros((32,96,96,3))
        for filename in filenames:
          img_path = os.path.join(path,filename)
          x_train = cv2.imread(img_path)
          x_trainx = cv2.resize(x_train,(24,24))
          x_trainy = cv2.resize(x_train,(96,96))
          x_train_x[i] = x_trainx 
          x_train_y[i] = x_trainy
          i = i+1
          if i == 32:
            j = j + 1
            print("batch {}/254".format(j))
            X_batch, Y_batch = x_train_x, x_train_y
            X_batch = tf.cast(X_batch, tf.float32)
            Y_batch = tf.cast(Y_batch, tf.float32)
            generated_images = generator(X_batch)
            X = tf.cast(generated_images, tf.float32) 
            X_fake_and_real = tf.concat([X, Y_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            model.train_on_batch(X_batch, y2)
            i = 0
            x_train_x = np.zeros((32,24,24,3))
            x_train_y = np.zeros((32,96,96,3))
    
      #Check result after every 100 epochs
      if (epoch+1)%100 == 0:

        generator.save_weights("Generator{}.h5".format(epoch))
        discriminator.save_weights("Discriminator_weights{}.h5".format(epoch))
        model.save_weights("Model{}.h5".format(epoch))
        from google.colab.patches import cv2_imshow

        path = "/content/drive/MyDrive/cars_train/07336.jpg"

        X = cv2.imread(path)
        X = cv2.resize(X,(24,24))
        X = np.reshape(X, (1,24,24,3))
        X_batch = tf.cast(X, tf.float32)

        Y = generator(X_batch)
        cv2_imshow(X[0])
        cv2_imshow( Y[0].numpy())



generator().summary()
discriminator().summary()
model = tf.keras.models.Sequential()
model.add(generator())
model.add(discriminator())
model.summary()
discriminator().compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator().trainable = False
model.compile(loss="binary_crossentropy", optimizer="rmsprop")

train_dcgan(model,epochs=2200)