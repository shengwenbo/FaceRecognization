import pandas as pd
import utils
from vgg16 import VGG16
from vgg11  import VGG11
import pickle
import pandas as pd
import gc
import  sys

if __name__ == "__main__":
    mode = sys.argv[1]
    data_from = sys.argv[2]
    data_path = sys.argv[3]

    if data_from == "image":
        data = utils.load_pie_jpg(data_path)
        with open("./data/PIE/data.pkl", "wb") as fout:
            pickle.dump(data, fout)

    else:
        with open("./data/PIE/data.pkl", "rb") as fin:
            data = pickle.load(fin)

    data = data.sample(frac=1).reset_index(drop=True)
    train_data = utils.get_train_data(data)
    test_data = utils.get_test_data(data)

    del data
    if mode == "train":
        module = VGG11(64, 64, batch_size=32,learning_rate=0.001, mode=mode)
        del test_data
        gc.collect()
        module.train(train_data[0], train_data[1])
    elif mode == "eval":
        module = VGG11(64, 64, batch_size=test_data[0].shape[0], mode=mode)
        del train_data
        gc.collect()
        module.eval(test_data[0], test_data[1])