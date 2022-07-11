import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import config

logging.basicConfig(level=logging.INFO, format='[ArTaxOr] %(process)d-%(levelname)s-%(message)s')
logging.info("TensorFlow Version:  " + tf.__version__)


data_url = config.data_url
logging.info(data_url)
data = tf.keras.utils.image_dataset_from_directory(data_url,
                                                   labels = 'inferred',
                                                   color_mode='rgb',
                                                   batch_size=config.batch_size,
                                                   image_size = config.image_size,
                                                   )
class_names = data.class_names
plt.figure(figsize=(10, 10))
for images, labels in data.take(1):
    print("Image shape: ", images.numpy().shape)
    print("Label: ", labels.numpy().shape)
    for i in range(0,4):
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()