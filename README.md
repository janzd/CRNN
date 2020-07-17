## CRNN

Keras implementation of Convolutional Recurrent Neural Network for text recognition

There are two models available in this implementation. One is based on the original CRNN model, and the other one includes a spatial transformer network layer to rectify the text. However, the performance does not differ very much, so it is up to you which model you choose.


### Training

You can use the Synth90k dataset to train the model, but you can also use your own data. If you use your own data, you will have to rewrite the code that loads the data accordingly to the structure of your data.
To download the Synth90k dataset, go to this [page](http://www.robots.ox.ac.uk/~vgg/data/text/) and download the MJSynth dataset.

Either put the Synth90k dataset in `data/Synth90k` or specify the path to the dataset using the `--base_dir` argument. The base directory should include a lot of subdirectories with Synth90k data, annotation files for training, validation, and test data, a file listing paths to all images in the dataset, and a lexicon file.

Use the `--model` argument to specify which of the two available models you want to use. The default model is CRNN with STN layer. See `config.py` for details.

Run the `train.py` script to perform training, and use the arguments accordingly to your setup.

#### Execution example

```
python train.py --batch_size 512 --gpus 0 1 2 3 --nb_workers 12
```

You can resume training by setting `--resume_training` to True and defining the path to the model you want to resume training, `--load_model_path`.

A pretrained model is available [here](https://drive.google.com/file/d/1hmtbUQC5HuLb1KOMozNwCKFoAPa56rtx/view?usp=sharing).


### Evaluation

Use `eval.py` to perform evaluation. You can either classify a single image or pass a directory with images that you want to classify. You also have to specify the path to a trained model.

#### Execution example

```
python eval.py --model_path result/001/model.hdf5 --data_path path/to/your/data
```

### Requirements

TensorFlow 1.X (tested with 1.5 but I think it should work with other 1.X versions too)
Keras 2.1.5 

The code uses a standalone Keras installation with TF backend. It does not use Keras included in TF.


