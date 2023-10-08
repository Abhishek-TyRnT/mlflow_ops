import tensorflow as tf
import time
import sys

class Train:
    def __init__(self,  model, loss_func, optimizer, apply_regularization = False):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.apply_regularization = apply_regularization


    def get_data_string(self, steps, current_step, loss):
        total_bar = 50
        increase_step = int(steps/total_bar + 0.5)
        for i in range(0,current_step+1):
            if i%increase_step == 0:
                dashes = int(i/increase_step + 0.5)
                dots   = total_bar - dashes
                string = '='*( dashes - 1)
                if not i == steps and not i == 0:
                    string += '>'
                    string += '.'*(dots - 1)

        data_String = '{0}/{1} :'.format(current_step,steps) + string
        data_String += ' Loss: ' + str(loss)
        return data_String

    @tf.function
    def train_one_step(self, inp, y_true):

        with tf.GradientTape() as tape:
            y_pred = self.model(inp)
            loss   = self.loss_func(y_true, y_pred)
            if self.apply_regularization:
                loss += tf.reduce_sum(self.model.losses)
            grads  = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss


    def fit(self, train_dataset, val_dataset, epochs, steps_per_epoch, val_steps):
        epoch = 0

        while epoch < epochs:
            step  = 0
            avg_loss = 0.
            print('Epoch :' + str(epoch))
            for img, _cls in train_dataset.take(steps_per_epoch):
                loss = self.train_one_step(img, _cls)
                loss       = float(loss)
                avg_loss   = (avg_loss*step + loss)/(step + 1)

                string = self.get_data_string(steps_per_epoch, step, avg_loss)
                sys.stdout.write('\r' + string)
                time.sleep(0.01)
                step += 1
            step = 0
            val_avg_loss = 0.

            for img, _cls in val_dataset.take(val_steps):
                y_true = _cls
                y_pred = self.model(img)
                loss   = self.loss_func(y_true,y_pred)
                val_avg_loss = (val_avg_loss * step + loss) / (step + 1)
                step += 1
            string += ' Val Loss: ' + str(val_avg_loss)
            sys.stdout.write('\r' + string)
            print()
            epoch += 1

        return self.model



