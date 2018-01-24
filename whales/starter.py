import matplotlib.image as mpimg
import pandas as pd
import numpy as np

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


data_folder = "/media/sree/mars/data/whales"

labels_file = data_folder + "/train.csv"
labels = pd.read_csv(labels_file)

unique_labels = pd.unique(labels['Id'])


# There are 4251 unique labels and we have 9850 images

# How is the distibution of labels?
idcount = labels.groupby(['Id'], as_index = False)["Image"].count()
idcount.sort_values('Image', inplace=True)


## filter those whales which are above 10 in count
images_to_consider = idcount.query('Image>10')

## move these images to another folder
new_folder = data_folder + "/filtered_train_dataset"


images = images_to_consider['Id'].tolist()
images = images[]
from shutil import copyfile

for image in images:
    name = data_folder + "/train/" + image + ".jpg"
    print(name)
    copyfile(name, new_folder)

images_to_consider['Id']

## see one of the images

