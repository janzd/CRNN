import os, glob
from keras.callbacks import ModelCheckpoint


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


