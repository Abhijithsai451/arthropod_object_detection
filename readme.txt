Used the tensorflow preprocessing methods to import the data
Found 15376 files belonging to 7 classes.
Using 12301 files for training.
Found 15376 files belonging to 7 classes.
Using 3075 files for validation.
Data belongs to the classes ->['Araneae', 'Coleoptera', 'Diptera', 'Hemiptera', 'Hymenoptera', 'Lepidoptera', 'Odonata']
image_dataset_from_directory() search for all images in the directory and import it.
While importing the images gets resized to the specified format i.e(128,128) in batches
When same seed is used we can divide the data directly into train and validataion set ## Visualizing the data
Using matplot lib, picked a single batch of data and processed it to create a subplot(4,4) ## Saving the preprocessed data to a location to enable the GPU
the dataset of format tf.data can be saved to a location using tf.data.experimental.save() function
Saved data can only be retrieved from tf.data.experimental.load() function so the dataset structure can be maintained