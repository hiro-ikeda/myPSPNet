from keras.applications import ResNet50
from keras.models import Model

from keras.layers import Input, Reshape, Dense, Conv2D, AveragePooling2D, concatenate, UpSampling2D, Activation, BatchNormalization, Lambda, Conv2DTranspose, Cropping2D
from keras.optimizers import Adam
from keras.backend import tf as ktf

import numpy as np

#bilinear interpolation
def Interp(x, shape) :
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(
            x,
            [int(new_height), int(new_width)],
            align_corners=True)
    return resized

#create model and return
def PSPNet50(input_shape, NUM_OF_CLASSES) :


    # num of filiters
    NUM_pyramid_filters = 512

    #this variable is needed to arrange your tasks
    feature_map_shape = (100, 100)

    input = Input(shape=input_shape)

    x = ResNet50(include_top=False)(input)

    Pyramid1 = AveragePooling2D(pool_size=(1,1), padding='same')(x)
    Pyramid1 = Conv2D(filters=NUM_pyramid_filters, kernel_size=(1,1))(Pyramid1)
    Pyramid1 = BatchNormalization(axis=3)(Pyramid1)
    Pyramid1 = Lambda(Interp, arguments={'shape' : feature_map_shape})(Pyramid1)

    Pyramid2 = AveragePooling2D(pool_size=(2,2), padding='same')(x)
    Pyramid2 = Conv2D(filters=NUM_pyramid_filters, kernel_size=(1,1))(Pyramid2)
    Pyramid2 = BatchNormalization(axis=3)(Pyramid2)
    Pyramid2 = Lambda(Interp, arguments={'shape' : feature_map_shape})(Pyramid2)

    Pyramid3 = AveragePooling2D(pool_size=(3,3), padding='same')(x)
    Pyramid3 = Conv2D(filters=NUM_pyramid_filters, kernel_size=(1,1))(Pyramid3)
    Pyramid3 = BatchNormalization(axis=3)(Pyramid3)
    Pyramid3 = Lambda(Interp, arguments={'shape' : feature_map_shape})(Pyramid3)

    Pyramid4 = AveragePooling2D(pool_size=(6,6), padding='same')(x)
    Pyramid4 = Conv2D(filters=NUM_pyramid_filters, kernel_size=(1,1))(Pyramid4)
    Pyramid4 = BatchNormalization(axis=3)(Pyramid4)
    Pyramid4 = Lambda(Interp, arguments={'shape' : feature_map_shape})(Pyramid4)

    y = concatenate([Pyramid1, Pyramid2, Pyramid3, Pyramid4, x], axis=3)

    y = Conv2D(filters=512, kernel_size=(1,1))(y)

    y = BatchNormalization(axis=3)(y)

    y = Activation('relu')(y)

    y = Conv2DTranspose(filters=NUM_OF_CLASSES, kernel_size=(32, 32),
                            strides=(32, 32), padding='same')(y)

    y = Cropping2D(((4, 4), (8, 8)))(y)

    y = BatchNormalization(axis=3)(y)

    out = Activation('softmax')(y)

    model = Model(inputs=input, outputs=out)

    return model
