import keras
import tensorflow as tf
import logging

from keras.layers import Conv2D, MaxPooling2D, Rescaling, Normalization, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.python.keras import Input

import config


logging.basicConfig(level=logging.INFO, format='[ArTaxOr] %(process)d-%(levelname)s-%(message)s')
logging.info("TensorFlow Version:  " + tf.__version__)


data_url = config.data_url
logging.info(data_url)
train_data = tf.keras.utils.image_dataset_from_directory(data_url,
                                                   labels = 'inferred',
                                                   color_mode='rgb',
                                                   batch_size=config.batch_size,
                                                   image_size = config.image_size,
                                                   subset='training',
                                                   seed=1337,
                                                   validation_split = 0.25,
                                                   )
val_data = tf.keras.utils.image_dataset_from_directory(data_url,
                                                   labels = 'inferred',
                                                   color_mode='rgb',
                                                   batch_size=config.batch_size,
                                                   image_size = config.image_size,
                                                   subset='validation',
                                                   validation_split =0.25,
                                                   seed=1337,
                                                   )

num_elements = tf.data.experimental.cardinality(val_data).numpy()
class_names = train_data.class_names
print("num_elements-> ",num_elements)
print("class_names--> ",class_names)

# Partitioning the data
def get_dataset_partitions_tf(data, data_size, val_split=0.5, test_split=0.5, shuffle=False):
    assert (val_split + test_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        data = data.shuffle(config.batch_size, seed=12)

    val_size = int(val_split * data_size)
    test_size = int(test_split * data_size)

    train_ds = data.take(val_size)
    test_ds = data.skip(val_size).take(test_size)

    return val_data, test_ds


logging.info("partitioning the validation data to test and validation splits")
#val_data, test_ds = get_dataset_partitions_tf(val_data, num_elements)
image_shape = ''
for image, label in train_data:
    data_shape = image.shape
    logging.info(image.shape)
    logging.info(label.shape)
    break

logging.info("partitioned the validation data to test and validation splits")

# Creating a Model

class create_model(tf.keras.Model):
    logging.info("Creating a model class")
    def __init__(self,image_shape, c):
        super().__init__()
        self.image_shape = image_shape
        self.num_classes = num_classes

        self.input_layer = Input(shape = image_shape)
        self.rescale = Rescaling(scale = 1/.255)
        self.conv1 = Conv2D(128, (3,3), input_shape=self.image_shape, activation='relu')
        self.conv2 = Conv2D(64, (3,3), input_shape=self.image_shape, activation='relu')
        self.conv3 = Conv2D(32, (3,3), input_shape=self.image_shape, activation='relu')
        self.conv4 = Conv2D(16, (3,3), input_shape=self.image_shape, activation='relu')
        self.conv5 = Conv2D(8, (3,3), input_shape=self.image_shape, activation='relu')
        self.maxpool = MaxPooling2D()
        self.Norm =  BatchNormalization()
        self.dense1 = Dense(8,activation='relu')
        self.dense2 = Dense(self.num_classes)
        self.flat = Flatten()
        self.dropout = Dropout(0.25)

    def call(self, input_data):
        x = self.input_layer(input_data)
        x = self.rescale(input_shape = (config.image_size, 3))(x)
        x = self.conv1()(x)
        x = self.maxpool()(x)
        x = self.conv2()(x)
        x = self.maxpool()(x)
        x = self.conv3()(x)
        x = self.maxpool()(x)
        x = self.conv4()(x)
        x = self.maxpool()(x)
        x = self.conv5()(x)
        x = self.maxpool()(x)
        x = self.flat(x)
        x= self.dropout()(x)
        x = self.dense1()
        logits = self.dense2()

        return logits

model = create_model(data_shape, len(class_names))
logging.info("Created the Model")
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
logging.info("Training the model on data: 'train_data' ")
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
epochs = 3
for epoch in range(epochs):
    with tf.GradientTape() as Tape:
        for step, (images, labels) in enumerate(train_data):

            y_pred = model(images)
            loss = keras.losses.mean_squared_error(labels,y_pred)
            gradients = Tape.gradient(loss,model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss))
                )
                print("Seen so far: %s samples" % ((step + 1) * config.batch_size))

