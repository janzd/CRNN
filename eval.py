import os
import argparse
import string
from tqdm import tqdm
import numpy as np
import cv2
import keras.backend as K
from keras.models import model_from_json, load_model

from utils import pad_image, resize_image, create_result_subdir
from STN.spatial_transformer import SpatialTransformer
from models import CRNN, CRNN_STN

parser = argparse.ArgumentParser()
parser.add_argument('--model_json_path', type=str, default='')
parser.add_argument('--model_weights_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--gpus', type=int, nargs='*', default=[0])
parser.add_argument('--characters', type=str, default='0123456789'+string.ascii_lowercase+'-')
parser.add_argument('--label_len', type=int, default=16)
parser.add_argument('--nb_channels', type=int, default=1)
parser.add_argument('--width', type=int, default=200)
parser.add_argument('--height', type=int, default=31)
parser.add_argument('--model', type=str, default='CRNN_STN', choices=['CRNN_STN', 'CRNN'])
parser.add_argument('--conv_filter_size', type=int, nargs=7, default=[64, 128, 256, 256, 512, 512, 512])
parser.add_argument('--lstm_nb_units', type=int, nargs=2, default=[128, 128])
parser.add_argument('--timesteps', type=int, default=50)
parser.add_argument('--dropout_rate', type=float, default=0.25)
cfg = parser.parse_args()

def set_gpus():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)[1:-1]

def create_output_directory():
    os.makedirs('eval', exist_ok=True)
    output_subdir = create_result_subdir('eval')
    print('Output directory: ' + output_subdir)
    return output_subdir

def collect_data():
    if os.path.isfile(cfg.data_path):
        return [cfg.data_path]
    else:
        files = [os.path.join(cfg.data_path, f) for f in os.listdir(cfg.data_path) if f[-4:] in ['.jpg', '.JPG', '.png', '.PNG']]
        return files

def load_image(img_path):
    if cfg.nb_channels == 1:
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(img_path)    

def preprocess_image(img):
    if img.shape[1] / img.shape[0] < 6.4:
        img = pad_image(img, (cfg.width, cfg.height), cfg.nb_channels)
    else:
        img = resize_image(img, (cfg.width, cfg.height))
    if cfg.nb_channels == 1:
        img = img.transpose([1, 0])
    else:
        img = img.transpose([1, 0, 2])
    img = np.flip(img, 1)
    img = img / 255.0
    if cfg.nb_channels == 1:
        img = img[:, :, np.newaxis]
    return img

def predict_text(model, img):
    y_pred = model.predict(img[np.newaxis, :, :, :])
    shape = y_pred[:, 2:, :].shape
    ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
    ctc_out = K.get_value(ctc_decode)[:, :cfg.label_len]
    result_str = ''.join([cfg.characters[c] for c in ctc_out[0]])
    result_str = result_str.replace('-', '')
    return result_str

def evaluate(model, data, output_subdir):
    if len(data) == 1:
        evaluate_one(model, data)
    else:
        evaluate_batch(model, data, output_subdir)

def evaluate_one(model, data):
    img = load_image(data[0])
    img = preprocess_image(img)
    result = predict_text(model, img)
    print('Detected result: {}'.format(result))

def evaluate_batch(model, data, output_subdir):
    for filepath in tqdm(data):        
        img = load_image(filepath)
        img = preprocess_image(img)
        result = predict_text(model, img)
        output_file = os.path.basename(filepath)
        output_file = output_file[:-4] + '.txt'
        with open(os.path.join(output_subdir, output_file), 'w') as f:
            f.write(result)

if __name__ == '__main__':
    set_gpus()
    output_subdir = create_output_directory()
    data = collect_data()
    _, model = CRNN_STN(cfg)    
    model.load_weights(cfg.model_weights_path)
    #model = load_model(cfg.model_weights_path, custom_objects={'SpatialTransformer': SpatialTransformer})
    evaluate(model, data, output_subdir)
