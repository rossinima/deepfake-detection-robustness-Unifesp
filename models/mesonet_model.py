# Este é o código-fonte oficial do MesoNet
# Nós não o escrevemos, apenas o usamos.
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras import backend as K

def Meso4(input_shape=None, **kwargs):
    """
    Cria a arquitetura do modelo Meso4
    """
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=256,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=False,
                                      weights=None)

    input_tensor = Input(shape=input_shape)
    x = input_tensor

    # Bloco 1
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Bloco 2
    x = Conv2D(8, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Bloco 3
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Bloco 4
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), padding='same')(x)

    # Flatten e Classificação
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(1, activation='sigmoid')(x) # 1 neurônio de saída (0=real, 1=fake)

    model = Model(input_tensor, x, name='meso4')
    return model