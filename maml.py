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
VAL_INTERVAL = 50
NUM_TRAIN_TASKS = 20
NUM_TEST_TASKS = 100
NUM_ITERATIONS = 1500

class MAML:
    def __init__(self, num_classes, num_inner_steps, inner_lr, outer_lr, 
                       n_support, n_query, log_dir:str='log'):
        """Initializes First-Order Model-Agnostic Meta-Learning.
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

        print("Initializing MAML model")
        model_layers = [layers.Input(shape=(28,28,1))]

        for i in range(NUM_CONV_LAYERS):
            model_layers.append(
                layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", name=f"Conv{i+1}")
            )
            model_layers.append(
                layers.BatchNormalization(name=f"BN{i+1}")
            )
            model_layers.append(
                layers.ReLU(name=f"ReLU{i+1}")
            )
        model_layers.append(layers.Flatten())
        model_layers.append(layers.Dense(num_classes, activation='softmax', name='Classification'))
        self.model = keras.Sequential(model_layers)
        self._log_dir = log_dir
        self._save_dir = os.path.join(log_dir, 'state')
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._save_dir, exist_ok=True)

        self._num_inner_steps = num_inner_steps
        self._inner_lr = inner_lr
        self._outer_lr = outer_lr
        self._optimizer = keras.optimizers.Adam(learning_rate=self._outer_lr)


        self.train_data = DataLoader('train', num_classes, n_support, n_query)
        self.val_data = DataLoader('test', num_classes, n_support, n_query)

        self._train_step = 0

        print("Finished initialization")

    def _inner_loop(self, theta, support_data):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            theta (List[Tensor]): current model parameters
            support_data (Tensor): task support set inputs
                images shape: (num_images, channels, height, width), labels shape: (num_images)

        Returns:
            parameters (List[Tensor]): adapted network parameters
            accuracies (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
        """

        accuracies = []

        phi = tf.nest.map_structure(lambda x: tf.Variable(tf.zeros_like(x)), self.model.trainable_weights)
        tf.nest.map_structure(lambda x, y: x.assign(y), phi, theta)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        opt_fn = tf.keras.optimizers.SGD(learning_rate=self._inner_lr)
        metrics_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='Inner Accuracy')

        ############### Your code here ###################
        # TODO: finish implementing this method.
        # This method computes the inner loop (adaptation) procedure for one
        # task. It also scores the model along the way.
        # Make sure to populate accuracies and update parameters.
        
        for _ in range(self._num_inner_steps + 1):
            for imgs, label in support_data:
                with tf.GradientTape() as tape:
                    logits = self.model(imgs, training=True)
                    loss = loss_fn(label, logits)
                grads = tape.gradient(loss, self.model.trainable_weights)
                opt_fn.apply_gradients(zip(grads, self.model.trainable_weights))
                opt_fn.apply_gradients(zip(grads, phi))
                metrics_fn.update_state(label, logits)
                accuracies.append(metrics_fn.result().numpy())
        #####################################################
        assert phi != None
        assert len(accuracies) == self._num_inner_steps + 1

        return phi, accuracies

    def _outer_loop(self, task_batch, train=None):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """

        theta = tf.nest.map_structure(lambda x: tf.Variable(tf.zeros_like(x)), self.model.trainable_weights)
        tf.nest.map_structure(lambda x, y: x.assign(y), theta, self.model.trainable_weights)

        metrics_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='Outer Accuracy')

        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []

        ############### Your code here ###################
        # TODO: finish implementing this method.
        # For a given task, use the _inner_loop method to adapt, then
        # compute the MAML loss and other metrics.
        # Make sure to populate outer_loss_batch, accuracies_support_batch,
        # and accuracy_query_batch.
        # Use keras.losses.SparseCategoricalCrossentropy to compute classification losses
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        
        for task in task_batch:
            support, query = task
            # support
            phi, accuracies = self._inner_loop(theta, support_data=support)
            # query
            query_loss = 0
            all_grads = tf.nest.map_structure(lambda x: tf.Variable(tf.zeros_like(x)), self.model.trainable_weights)
            for imgs, label in query:
                with tf.GradientTape() as tape:
                    logits = self.model(imgs, training=train)
                    loss = loss_fn(label, logits)
                
                grads = tape.gradient(loss, self.model.trainable_weights)
                all_grads = tf.nest.map_structure(lambda x, y: x + y, all_grads, grads)
                query_loss += loss
                metrics_fn.update_state(label, logits)

            accuracies_support_batch.append(accuracies)
            accuracy_query_batch.append(metrics_fn.result().numpy())
            outer_loss_batch.append(query_loss)

        self._optimizer.apply_gradients(zip(all_grads, theta))

        #####################################################
        
        # Update model with new theta
        tf.nest.map_structure(lambda x, y: x.assign(y), self.model.trainable_weights, theta)

        outer_loss = tf.reduce_mean(outer_loss_batch).numpy()
        accuracies_support = np.mean(accuracies_support_batch, axis=0)
        accuracy_query = np.mean(accuracy_query_batch)

        return outer_loss, accuracies_support, accuracy_query

    def train(self, train_steps):
        """Train the MAML.

        Optimizes MAML meta-parameters
        while periodically validating on validation_tasks, logging metrics, and
        saving checkpoints.

        Args:
            train_steps (int) : the number of steps this model should train for
        """
        print(f"Starting MAML training at iteration {self._train_step}")
        val_batches = self.val_data.generate_task(NUM_TEST_TASKS) 
        for i in range(1, train_steps+1):
            self._train_step += 1
            train_task = self.train_data.generate_task(NUM_TRAIN_TASKS)

            outer_loss, accuracies_support, accuracy_query = self._outer_loop(train_task, train=True)

            if self._train_step % SAVE_INTERVAL == 0:
                self._save_model()

            if i % LOG_INTERVAL == 0:
                print(
                    f'Iteration {self._train_step}: '
                    f'loss: {outer_loss:.3f} | '
                    f'support accuracy: '
                    f'{accuracies_support[-1]:.3f} | '
                    f'query accuracy: '
                    f'{accuracy_query:.3f}'
                )

                tf.summary.scalar('loss/train', outer_loss, self._train_step)
                tf.summary.scalar(
                    'train_accuracy/pre_adapt_support',
                    accuracies_support[0],
                    self._train_step
                )
                tf.summary.scalar(
                    'train_accuracy/post_adapt_support',
                    accuracies_support[-1],
                    self._train_step
                )
                tf.summary.scalar(
                    'train_accuracy/post_adapt_query',
                    accuracy_query,
                    self._train_step
                )

            if i % VAL_INTERVAL == 0:
                outer_loss, accuracies_support, accuracy_query = self._outer_loop(val_batches, train=False)
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
            _, _, accuracy_query = self._outer_loop(test_data, train=False)
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
        print("Loading checkpoint from", target_path)
        try:
            self.model = keras.models.load_model(target_path)
            self._train_step = checkpoint_step
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        except:
            raise ValueError(f'No checkpoint for iteration {checkpoint_step} found.')

    def _save_model(self):
        # Save a model to 'save_dir'
        self.model.save(os.path.join(self._save_dir, f"{self._train_step}.h5"))
        print("Saved Checkpoint")


def main(args):
    log_dir = args.log_dir
    if log_dir is None: log_dir = os.path.join(os.path.abspath('.'), 'p1_log')
    print(f'log_dir: {log_dir}')

    maml = MAML(
        args.num_way,
        args.num_inner_steps,
        args.inner_lr,
        args.outer_lr,
        args.num_support,
        args.num_query,
        log_dir
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    # Run "tensorboard --logdir [PATH TO LOG DIR]" to visualize graph
    callback = tf.keras.callbacks.TensorBoard(log_dir)
    callback.set_model(maml.model)
    writer = tf.summary.create_file_writer(log_dir)
    writer.set_as_default()

    if not args.test:
        print(
            f'Training with composition: \n'
            f'\tnum_way={args.num_way}\n'
            f'\tnum_support={args.num_support}\n'
            f'\tnum_query={args.num_query}'
        )
        maml.train(args.num_train_iterations)

    else:
        print(
            f'Testing with composition: \n'
            f'\tnum_way={args.num_way}\n'
            f'\tnum_support={args.num_support}\n'
            f'\tnum_query={args.num_query}'
        )

        maml.test()

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
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    with tf.device('/device:GPU:0'):
        main(args)

