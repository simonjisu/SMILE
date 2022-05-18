import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import random
import numpy as np
import os
import argparse

SEED = 1000
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from dataloader import DataLoader



NUM_CONV_LAYERS = 4
SAVE_INTERVAL = 100
LOG_INTERVAL = 1
VAL_INTERVAL = 5
NUM_TRAIN_TASKS = 20
NUM_TEST_TASKS = 100
NUM_ITERATIONS = 1000


class ConvLayer(layers.Layer):
    def __init__(self, filters, kernel_size, padding: str = 'same'):
        super(ConvLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, strides=2, padding=self.padding)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvNet(keras.Model):
    def __init__(self, classes=964, shape=(28,28,1)):
        super(ConvNet, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=shape),
            ConvLayer(64, 3, 'same'),
            ConvLayer(64, 3, 'same'),
            ConvLayer(64, 3, 'same'),
            ConvLayer(64, 3, 'same'),
            layers.Flatten()
        ])

        self.classification = layers.Dense(classes, activation='softmax')

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.classification(x)
        return x

class TransferLearning:
    def __init__(self, num_classes, num_inner_steps, inner_lr, outer_lr, 
                       n_support, n_query, log_dir:str='log'):
        """Initializes Transfer Learning vs. Meta-Learning.
        Model architecture from https://arxiv.org/abs/1703.03400
        The model consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and a ReLU activation.

        Args:
            num_classes (int): the number of classes in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
            outer_lr (float): learning rate for outer-loop optimization
            n_support (int): the number of support images in a task
            n_query (int): the number of query images in a task
            log_dir (str): path to logging metrics
        """

        print("Initializing convolutional model")
        self.model = ConvNet()
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=outer_lr), 
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['Accuracy']
        )

        self._log_dir = log_dir
        self._save_dir = os.path.join(log_dir, 'state')
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._save_dir, exist_ok=True)

        self._num_classes = num_classes
        self._num_inner_steps = num_inner_steps
        self._inner_lr = inner_lr
        self._outer_lr = outer_lr


        self.train_data = DataLoader('train', num_classes, n_support, n_query)
        self.val_data = DataLoader('test', num_classes, n_support, n_query)

        self._train_step = 0

        print("Finished initialization")

    def _transfer_learn(self, task_batch):
        """Creates, trains, and validates a new model using the pretrained feature extractor layers.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader

        Returns:
            outer_loss (Tensor): mean loss over the batch
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """

        feature_extractor = self.model.encoder
        feature_extractor.trainable = False

        opt_fn = tf.keras.optimizers.SGD(learning_rate=self._inner_lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []

        ############### Your code here ###################
        # TODO: finish implementing this method.
        # For a given task, create a new model with a new Dense classification layer,
        # use the model.fit method to adapt, then
        # compute the loss and other metrics.
        # Make sure to populate outer_loss_batch, accuracies_support_batch,
        # and accuracy_query_batch.

        for task in task_batch:
            support, query = task
            

        #####################################################
        outer_loss = tf.reduce_mean(outer_loss_batch).numpy()
        accuracies_support = np.mean(accuracies_support_batch, axis=0)
        accuracy_query = np.mean(accuracy_query_batch)

        return outer_loss, accuracies_support, accuracy_query

    def train(self, train_steps):
        print(f"Starting Transfer Learning training at iteration {self._train_step}")
        dataset = self.train_data.generate_entire_dataset()

        for i in range(1, train_steps+1):
            self._train_step += 1

            metrics = self.model.fit(dataset, epochs=1, verbose=0)
            train_loss, train_acc = metrics.history['loss'][-1], metrics.history['Accuracy'][-1]

            if self._train_step % SAVE_INTERVAL == 0:
                self._save_model()

            if i % LOG_INTERVAL == 0:
                print(
                    f'Iteration {self._train_step}: '
                    f'loss: {train_loss:.3f} | '
                    f'train accuracy: '
                    f'{train_acc:.3f}'
                )

                tf.summary.scalar('loss/train', train_loss, self._train_step)
                tf.summary.scalar(
                    'train_accuracy/train_all',
                    train_acc,
                    self._train_step
                )


            if i % VAL_INTERVAL == 0:
                val_dataset = self.val_data.generate_task()
                outer_loss, accuracies_support, accuracy_query = self._transfer_learn(val_dataset)
                loss= outer_loss
                accuracy_post_adapt_support = (accuracies_support[-1])
                accuracy_post_adapt_query = (accuracy_query)

                print(
                    f'\t-Validation: '
                    f'loss: {loss:.3f} | '
                    f'post-adaptation support accuracy: '
                    f'{accuracy_post_adapt_support:.3f} | '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_post_adapt_query:.3f}'
                )

                tf.summary.scalar('loss/validation', outer_loss, self._train_step)
                tf.summary.scalar(
                    'validation_accuracy/pre_adapt_support',
                    accuracies_support[0],
                    self._train_step
                )
                tf.summary.scalar(
                    'validation_accuracy/post_adapt_support',
                    accuracies_support[-1],
                    self._train_step
                )
                tf.summary.scalar(
                    'validation_accuracy/post_adapt_query',
                    accuracy_post_adapt_query,
                    self._train_step
                )

    def test(self):
        accuracies = []
        test = [self.val_data.generate_task(NUM_TEST_TASKS//10) for _ in range(10)]
        for test_data in test:
            _, _, accuracy_query = self._transfer_learn(test_data)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(10)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )
    
    def load(self, checkpoint_step):
        # Loads model from checkpoint step
        target_path = os.path.join(self._save_dir, f"{checkpoint_step}.h5")
        try:
            self.model = keras.models.load_model(target_path)
            self._train_step = checkpoint_step
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        except:
            raise ValueError(f'No checkpoint for iteration {checkpoint_step} found.')

    def _save_model(self):
        # Save a model to 'save_dir'
        self.model.save_weights(os.path.join(self._save_dir, f"{self._train_step}.h5"))
        print("Saved Checkpoint")


def main(args):
    log_dir = args.log_dir
    if log_dir is None: log_dir = os.path.join(os.path.abspath('.'), 'p2_log')
    print(f'log_dir: {log_dir}')

    transfer_model = TransferLearning(
        args.num_way,
        args.num_inner_steps,
        args.inner_lr,
        args.outer_lr,
        args.num_support,
        args.num_query,
        log_dir
    )

    if args.checkpoint_step > -1:
        transfer_model.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    # Run "tensorboard --logdir [PATH TO LOG DIR]" to visualize graph
    callback = tf.keras.callbacks.TensorBoard(log_dir)
    callback.set_model(transfer_model.model)
    writer = tf.summary.create_file_writer(log_dir)
    writer.set_as_default()

    if not args.test:
        print(
            f'Training with composition: \n'
            f'\tnum_way={args.num_way}\n'
            f'\tnum_support={args.num_support}\n'
            f'\tnum_query={args.num_query}'
        )
        transfer_model.train(args.num_train_iterations)

    else:
        print(
            f'Testing with composition: \n'
            f'\tnum_way={args.num_way}\n'
            f'\tnum_support={args.num_support}\n'
            f'\tnum_query={args.num_query}'
        )

        transfer_model.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=5,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--num_inner_steps', type=int, default=5,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--num_train_iterations', type=int, default=500,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    args = parser.parse_args()
    main(args)

