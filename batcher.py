import tensorflow as tf
from queue import Queue

class Batcher(object):
    """
    images: [total_count, height, width, channels]
    labels: [total_count]
    """
    def __init__(self, images, labels,batch_size=32):
        self.image_que = Queue()
        self.label_que = Queue()
        self.batch_size = batch_size
        self._batch_count = 0
        self._cur_batch = 0
        self._fill_data(images, labels)

    def _fill_data(self, images, labels):
        index = 0
        while True:
            if index+self.batch_size > images.shape[0]:
                self.image_que.put(images[index:-1, :, :, :])
                self.label_que.put(labels[index:-1])
                self._batch_count += 1
                return
            self.image_que.put(images[index:index+self.batch_size,:,:,:])
            self.label_que.put(labels[index:index+self.batch_size])
            index += self.batch_size
            self._batch_count += 1

    def next_batch(self):
        image_batch = self.image_que.get()
        self.image_que.put(image_batch)
        label_batch = self.label_que.get()
        self.label_que.put(label_batch)
        self._cur_batch += 1
        finished = self._cur_batch == self._batch_count
        if finished:
            self._cur_batch = 0
        return self.image_que.get(), self.label_que.get(),finished