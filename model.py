import tensorflow as tf
import logging
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
val_data, test_ds = get_dataset_partitions_tf(val_data, num_elements)
for image, label in test_ds:
    logging.info(image.shape)
    logging.info(label.shape)
    break

logging.info("partitioned the validation data to test and validation splits")

# Creating a Model

