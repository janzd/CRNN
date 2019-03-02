from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.utils import multi_gpu_model

from models import CRNN_STN, CRNN
from utils import MultiGPUModelCheckpoint, PredictionModelCheckpoint, Evaluator


def set_gpus():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)[1:-1]

def create_output_directory():
    output_subdir = create_result_subdir(cfg.output_dir)
    print('Output directory: ' + output_subdir)
    return output_subdir

def get_models():
	if cfg.model == 'CRNN_STN':
        return CRNN_STN(cfg)
    else
        return CRNN(cfg)

def get_optimizer():
	if cfg.optimizer = 'sgd':
		opt = SGD(lr=cfg.lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
	elif cfg.optimizer = 'adam':
		opt = Adam(lr=cfg.lr)
	else:
		raise ValueError('Wrong optimizer name')
	return opt

def get_callbacks(output_subdir, training_model, prediction_model):
	training_model_checkpoint = MultiGPUModelCheckpoint(os.path.join(output_subdir, cfg.training_model_cp_filename), training_model, save_best_only=cfg.save_best_only, monitor='loss', mode='min')
	prediction_model_checkpoint = PredictionModelCheckpoint(os.path.join(output_subdir, cfg.prediction_model_cp_filename), prediction_model, save_best_only=cfg.save_best_only, monitor='loss', mode='min')
	evaluator = Evaluator(prediction_model)
	lr_reducer = ReduceLROnPlateau(factor=cfg.lr_reduction_factor, patience=3, verbose=1, min_lr=0.00001)
	tensorboard = TensorBoard(log_dir=cfg.tb_log)
	return [training_model_checkpoint, prediction_model_checkpoint, evaluator, lr_reducer, tensorboard]

def load_weights_if_resume_training(training_model):
	if cfg.resume_training:
		assert cfg.load_model_path != '', 'Path to model which you want to resume training is not declared!'
	    training_model.load_weights(cfg.load_model_path)
	return training_model

def instantiate_multigpu_model_if_multiple_gpus(training_model):
	if len(cfg.gpus) > 1:
		training_model = multi_gpu_model(training_model, len(cfg.gpus))
	return training_model

if __name__ == '__main__':
	set_gpus()
	output_subdir = create_output_directory()
    training_model, prediction_model = get_models()
    training_model.summary()
    opt = get_optimizer()
    callbacks = get_callbacks(output_subdir, training_model, prediction_model)
    training_model = load_weights_if_resume_training()
    training_model = instantiate_multigpu_model_if_multiple_gpus()
    training_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)
    print(model.optimizer)
    print("Learning rate: " + str(K.eval(model.optimizer.lr)))
    training_model.fit_generator(train_generator, steps_per_epoch=int(file_list_len / cfg.batch_size), epochs=cfg.nb_epochs, verbose=1, workers=cfg.nb_workers, use_multiprocessing=True, callbacks=callbacks)

