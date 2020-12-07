from util import resolve_single, resolve, normalize, denormalize, normalize_01, normalize_m11, denormalize_m11, pixel_shuffle
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, \
    Lambda, Activation, Concatenate, Multiply, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19

LR_SIZE = 24
HR_SIZE = 96
upscaling_factor = 4
channels = 3
filters = 64


def SubpixelConv2D(name, scale=2):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                None if input_shape[1] is None else input_shape[1] * scale,
                None if input_shape[2] is None else input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)


def upsample(x, number):
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_' + str(number))(x)
    x = SubpixelConv2D(name=str('upSampleSubPixel_') + str(number), scale=2)(x)
    x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
    return x


def dense_block(input):
    x1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Concatenate()([input, x1])

    x2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x1)
    x2 = LeakyReLU(0.2)(x2)
    x2 = Concatenate()([input, x1, x2])

    x3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x2)
    x3 = LeakyReLU(0.2)(x3)
    x3 = Concatenate()([input, x1, x2, x3])

    x4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x3)
    x4 = LeakyReLU(0.2)(x4)
    x4 = Concatenate()([input, x1, x2, x3, x4])

    x5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x4)
    x5 = Lambda(lambda x: x * 0.2)(x5)
    x = Add()([x5, input])
    return x


def RRDB(input):
    x = dense_block(input)
    x = dense_block(x)
    x = dense_block(x)
    x = Lambda(lambda x: x * 0.2)(x)
    out = Add()([x, input])
    return out


def sr_resnet(num_filters=64, num_res_blocks=16):
    lr_input = Input(shape=(24, 24, 3))

    x_start = Conv2D(64, kernel_size=3, strides=1, padding='same')(lr_input)
    x_start = LeakyReLU(0.2)(x_start)

    x = RRDB(x_start)

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = Lambda(lambda x: x * 0.2)(x)
    x = Add()([x, x_start])

    x = upsample(x, 1)
    if upscaling_factor > 2:
        x = upsample(x, 2)
    if upscaling_factor > 4:
        x = upsample(x, 3)

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    hr_output = Conv2D(channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

    model = Model(inputs=lr_input, outputs=hr_output)

    return model


generator = sr_resnet


def conv2d_block(input, filters, strides=1, bn=True):
    d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    return d


def discriminator(num_filters=64):
    img = Input(shape=(HR_SIZE, HR_SIZE, 3))
    x = conv2d_block(img, filters, bn=False)
    x = conv2d_block(x, filters, strides=2)
    x = conv2d_block(x, filters * 2)
    x = conv2d_block(x, filters * 2, strides=2)
    x = conv2d_block(x, filters * 4)
    x = conv2d_block(x, filters * 4, strides=2)
    x = conv2d_block(x, filters * 8)
    x = conv2d_block(x, filters * 8, strides=2)
    x = Dense(filters * 16)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(1)(x)

    # Create model and compile
    model = Model(inputs=img, outputs=x)
    return model