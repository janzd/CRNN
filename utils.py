import os, glob
import numpy as np
import cv2

from keras.callbacks import ModelCheckpoint, Callback
import keras.backend as K


def create_result_subdir(result_dir):

    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase)
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d' % (run_id))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    return result_subdir


class MultiGPUModelCheckpoint(ModelCheckpoint):

    def __init__(self, filepath, alternate_model, **kwargs):
        """
        Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are accepted.
        :param filepath:
        :param alternate_model: Keras model to save instead of the default. This is used especially when training multi-
                                gpu models built with Keras multi_gpu_model(). In that case, you would pass the original
                                "template model" to be saved each checkpoint.
        :param kwargs:          Passed to ModelCheckpoint.
        """

        self.alternate_model = alternate_model
        super().__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        model_before = self.model
        self.model = self.alternate_model
        super().on_epoch_end(epoch, logs)
        self.model = model_before


class PredictionModelCheckpoint(Callback):

    def __init__(self, filepath, prediction_model, monitor='loss', save_best_only=False, mode='min', period=1, save_weights_only=False, verbose=False):
        self.filepath = filepath
        self.prediction_model = prediction_model
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.period = period
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.epochs_since_last_save = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.prediction_model.save_weights(filepath, overwrite=True)
                        else:
                            self.prediction_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.prediction_model.save_weights(filepath, overwrite=True)
                else:
                    self.prediction_model.save(filepath, overwrite=True)


class Evaluator(Callback):

    def __init__(self, prediction_model, val_generator, label_len, characters, optimizer, period=2000):
        self.prediction_model = prediction_model
        self.period = period
        self.val_generator = val_generator
        self.label_len = label_len
        self.characters = characters
        self.optimizer = optimizer

    def on_batch_end(self, batch, logs=None):
        if ((batch+1) % self.period) == 0:
            accuracy, correct_char_predictions = self.evaluate()
            print('=====================================')
            print('Word level accuracy: %.3f' % accuracy)
            print('Correct character level predictions: %d' % correct_char_predictions)
            print('=====================================')

    def on_epoch_end(self, epoch, logs=None):
        accuracy, correct_char_predictions = self.evaluate()
        print('=====================================')
        print('After epoch %d' % epoch)
        print('Word level accuracy: %.3f' % accuracy)
        print('Correct character level predictions: %d' % correct_char_predictions)        
        if self.optimizer == 'sgd':
            lr = self.model.optimizer.lr
            decay = self.model.optimizer.decay
            iterations = self.model.optimizer.iterations
            lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
            print("Decayed learning rate: %.8f" % K.eval(lr_with_decay))
        else:
            print("Learning rate: %.8f" % K.eval(self.model.optimizer.lr))

    def evaluate(self):
        correct_predictions = 0
        correct_char_predictions = 0       

        x_val, y_val = self.val_generator[np.random.randint(0, int(self.val_generator.nb_samples / self.val_generator.batch_size))]
        #x_val, y_val = next(self.val_generator)

        y_pred = self.prediction_model.predict(x_val)

        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
        ctc_out = K.get_value(ctc_decode)[:, :self.label_len]

        for i in range(self.val_generator.batch_size):
            print(ctc_out[i])
            result_str = ''.join([self.characters[c] for c in ctc_out[i]])            
            result_str = result_str.replace('-', '')
            if result_str == y_val[i]:
                correct_predictions += 1
            print(result_str, y_val[i])
            
            for c1, c2 in zip(result_str, y_val[i]):
                if c1 == c2:
                    correct_char_predictions += 1

        return correct_predictions / self.val_generator.batch_size, correct_char_predictions



def pad_image(img, img_size, nb_channels):
    # img_size : (width, height)
    # loaded_img_shape : (height, width)
    img_reshape = cv2.resize(img, (int(img_size[1] / img.shape[0] * img.shape[1]), img_size[1]))
    if nb_channels == 1:
        padding = np.zeros((img_size[1], img_size[0] - int(img_size[1] / img.shape[0] * img.shape[1])), dtype=np.int32)
    else:
        padding = np.zeros((img_size[1], img_size[0] - int(img_size[1] / img.shape[0] * img.shape[1]), nb_channels), dtype=np.int32)
    img = np.concatenate([img_reshape, padding], axis=1)
    return img

def resize_image(img, img_size):
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    img = np.asarray(img)
    return img

