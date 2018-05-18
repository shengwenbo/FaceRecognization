import pandas as pd
import utils
from vgg16 import VGG16
import pickle
import pandas as pd

if __name__ == "__main__":
    # data = utils.load_pie_jpg("C:/images")
    # with open("./data/PIE/data.pkl", "wb") as fout:
    #     pickle.dump(data, fout)
    with open("./data/PIE/data.pkl", "rb") as fin:
        data = pickle.load(fin)
    train_data = utils.get_train_data(data)
    test_data = utils.get_test_data(data)

    module = VGG16(64,64)
    module.train(train_data[0], train_data[1])