import tensorflow as tf
import numpy as np
from model import Prototypical_Network
from tqdm import tqdm

def train_model(train_dataset, val_dataset, n_tasks:int, n_epochs:int=20, n_tpe:int=100, is_random:bool=False):
    @tf.function
    def loss_func(support, query):
        loss, acc = model(support, query)
        return loss, acc

    @tf.function
    def train(support, query):
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss, acc = model(support, query)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        # Log loss and accuracy for step
        train_loss(loss)
        train_acc(acc)

    @tf.function
    def validate(support, query):
        loss, acc = loss_func(support, query)
        val_loss(loss)
        val_acc(acc)

    def on_start_epoch(epoch):
        # print(f'Epoch{epoch + 1}')
        train_loss.reset_states()
        val_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states() 

    def on_end_epoch(train_loss, train_acc, val_loss, val_acc, val_losses, val_accs, train_losses, train_accs, progress_bar):
        # print(f'\t-Train Loss:{train_loss.result():.3f} | Train Acc:{train_acc.result() * 100:.2f}%\n\t-Val Loss:{val_loss.result():.3f} | Val Acc:{val_acc.result() * 100:.2f}%')
        s = f'Train Loss:{train_loss.result():.3f} | Train Acc:{train_acc.result() * 100:.2f}% | Val Loss:{val_loss.result():.3f} | Val Acc:{val_acc.result() * 100:.2f}%'
        progress_bar.set_postfix_str(s)

        train_losses.append(train_loss.result().numpy())
        train_accs.append(train_acc.result().numpy())

        val_losses.append(val_loss.result().numpy())
        val_accs.append(val_acc.result().numpy())
        
    model = Prototypical_Network()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)


    # Metrics to gather
    train_loss = tf.metrics.Mean(name='train_loss')
    val_loss = tf.metrics.Mean(name='val_loss')
    train_acc = tf.metrics.Mean(name='train_accuracy')
    val_acc = tf.metrics.Mean(name='val_accuracy')
    train_accs, train_losses = [],[]
    val_accs, val_losses = [],[]

    train_dataset.generate_task_list(n_tasks)

    progress_bar = tqdm()
    for epoch in range(n_epochs):
        on_start_epoch(epoch)
        multiplier = n_tpe // n_tasks if n_tpe > n_tasks else 1
        tasks = np.repeat(np.random.permutation(range(n_tasks)), multiplier)
        
        progress_bar.set_description(f'Training Epoch: {epoch+1}')
        progress_bar.reset(len(tasks[:n_tpe]))
        progress_bar.refresh()

        for task in tasks[:n_tpe]:
            if is_random:
                support, query = train_dataset.random_data_generator()
            else:
                support, query = train_dataset.data_generator(task)
            val_support, val_query = val_dataset.data_generator()

            train(support, query)
            validate(val_support, val_query)
            progress_bar.update()
            
        on_end_epoch(train_loss, train_acc, val_loss, val_acc, val_losses, val_accs, train_losses, train_accs, progress_bar)
        
        progress_bar.refresh()
    progress_bar.clear()

    return train_accs, train_losses, val_accs, val_losses

