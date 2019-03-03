import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--width')
parser.add_argument('--height')
parser.add_argument('--nb_channels')
parser.add_argument('--label_len')
parser.add_argument('--batch_size')
parser.add_argument('--grayscale')
parser.add_argument('--optimizer')
parser.add_argument('--lr')
parser.add_argument('--lr_reduction_factor')
parser.add_argument('--gpus')
parser.add_argument('--resume_training')

parser.add_argument('--characters')
parser.add_argument('--nb_classes')

#parser.add_argument('--lexicon_path')
#parser.add_argument('--img_folder')
parser.add_argument('--base_dir')

parser.add_argument('--output_dir')
parser.add_argument('--training_model_cp_filename')
parser.add_argument('--prediction_model_cp_filename')
parser.add_argument('--load_model_path')
parser.add_argument('--tb_log')