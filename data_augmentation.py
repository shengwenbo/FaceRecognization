import tensorflow as tf
import utils
from PIL import Image
import os
import collections

if __name__ == "__main__":

    counter = collections.Counter()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')

    data = utils.load_pie()
    for index,row in data.iterrows():
        image = row["fea"]
        label = row["gnd"]
        isTest = row["isTest"]
        image = image.reshape([1,64,64,1])

        if isTest < 0.5:
            counter.update([label])
            count = counter.get(label)
        else:
            count = 0
        fname = "%d_%s_%d" % (label, ("train" if isTest < 0.5 else "test"), count)

        pil_image = Image.fromarray(image.reshape(64,64),"L")
        pil_image.save("./data/PIE/images/" + fname+".jpg")

        if isTest > 0.5:
            continue

        i = 0
        for batch in datagen.flow(image, batch_size=1, save_to_dir="./data/PIE/images", save_prefix=fname, save_format="jpg"):
            i += 1
            if i > 0:
                break