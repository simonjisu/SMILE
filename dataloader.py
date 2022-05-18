import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np
import random

class DataLoader(object):
    def __init__(self, data_type:str='train', n_way:int=5, n_support:int=5, n_query:int=15):
        self.data_type = data_type
        self.data = self.preprocess_data(tfds.load("omniglot", data_dir='./data', split=self.data_type, as_supervised=True))
        self.labels = list(self.data.keys())
        
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.task_list = None
        
    def preprocess_data(self, dataset):
        print(f"\t-Preprocessing {self.data_type} Omniglot dataset")
        def preprocess(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28,28])
            return image, label

        self.dataset = dataset.map(preprocess)

        data = {}
        for image, label in self.dataset:
            image = image.numpy()
            label = str(label.numpy())
            if label not in data:
                data[label] = []
            data[label].append(image)
        return data
            
    def generate_mini_dataset(self):
        # Generates a support/query dataset for use in MAML

        support_images = np.zeros(shape=(self.n_way * self.n_support, 28, 28, 1))
        support_labels = np.zeros(shape=(self.n_way * self.n_support))

        query_images = np.zeros(shape=(self.n_way * self.n_query, 28, 28, 1))
        query_labels = np.zeros(shape=(self.n_way * self.n_query))

        # Get a random subset of labels from data.keys()
        labels = np.random.choice(self.labels, self.n_way, replace=False)

        for idx, label in enumerate(labels):
            support_labels[idx * self.n_support: (idx+1)*self.n_support] = idx
            query_labels[idx * self.n_query: (idx+1)*self.n_query] = idx

            images = random.choices(self.data[label], k=self.n_query + self.n_support)
            support_images[idx * self.n_support : (idx+1) * self.n_support] = images[:self.n_support]
            query_images[idx * self.n_query: (idx+1) * self.n_query] = images[-self.n_query:]

        support_dataset = tf.data.Dataset.from_tensor_slices(
            (support_images.astype(np.float32), support_labels.astype(np.int8))
        )
        query_dataset = tf.data.Dataset.from_tensor_slices(
            (query_images.astype(np.float32), query_labels.astype(np.int8))
        )

        support_dataset = support_dataset.shuffle(100).batch(25, drop_remainder=False)
        query_dataset = query_dataset.shuffle(100).batch(25, drop_remainder=False)
        
        return support_dataset, query_dataset

    def generate_task(self, num_batch_per_task:int= 10):
        return [self.generate_mini_dataset() for _ in range(num_batch_per_task)]
        
    def generate_entire_dataset(self):
        def preprocess_labels(image, label):
            label = tf.one_hot(label, depth=len(self.data.keys()), dtype=tf.int8)
            return image, label
        dataset = self.dataset.map(preprocess_labels)
        dataset = dataset.shuffle(100).batch(25)

        return dataset