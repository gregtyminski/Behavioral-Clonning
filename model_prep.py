import tensorflow as tf
from tensorflow.keras.layers import Lambda, Cropping2D, Dense, GlobalAveragePooling2D, Flatten, ZeroPadding2D, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

from DataGenerator import DataGenerator

def saturation_converter(x):
    hsv = tf.image.rgb_to_hsv(x)
    return hsv[: , : , : , :1: ]


model = Sequential([
    Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3), name='normalize'),
    Cropping2D(cropping=((65,25), (0,0)), name='cropping_65_25'),
    Lambda(saturation_converter, name='saturation'),
    BatchNormalization(),
    Conv2D(8,(3,3),padding='valid', activation='relu'),
    BatchNormalization(),
    Conv2D(16,(3,3),padding='valid', activation='relu'),
    BatchNormalization(),
    Conv2D(32,(3,3),padding='valid', activation='relu'),
    GlobalAveragePooling2D(),
    Dense(16, activation='linear', name='dense-64'),
    Dropout(0.5),
    Dense(1, activation='linear', name='dense-1')
])
model.summary()

epochs = 120
patience = 10
batch_size = 64
learn_rate = 0.0001

optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
model.compile(loss='mse', optimizer=optimizer)
generator = DataGenerator('./udacity_data/', epochs=epochs, batch_size=batch_size, balance=True, debug=False)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience, min_lr=1e-6, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience+1)
fit_result = model.fit_generator(generator,
                                 steps_per_epoch=generator.batches,
                                 epochs=generator.epochs,
                                 verbose=1,
                                 callbacks=[reduce_lr, early_stop])

generator = DataGenerator('./my_data/', epochs=epochs, batch_size=batch_size, balance=True, debug=False)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience, min_lr=1e-6, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience+1)
fit_result = model.fit_generator(generator,
                                 steps_per_epoch=generator.batches,
                                 epochs=generator.epochs,
                                 verbose=1,
                                 callbacks=[reduce_lr, early_stop])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")