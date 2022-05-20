from pathlib import Path
import tensorflow_datasets as tfds

data_dir = Path('./data/')
ds = tfds.load("omniglot", data_dir=data_dir, as_supervised=True)