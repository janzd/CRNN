## CRNN

Keras implementation of Convolutional Recurrent Neural Network for text recognition




### Training

You can use the Synth90k dataset to train the model, but you can also use your own data. If you use your own data, some of the hyperparameters might not be optimal and you might want to try to find better values.
To download the Synth90k dataset, go to this [page|http://www.robots.ox.ac.uk/~vgg/data/text/] and download the MJSynth dataset.

Either put the Synth90k dataset in `data/Synth90k` or specify the path to the dataset using the `--base_dir` argument. 

Run the `train.py` script to perform training, and use the arguments accordingly to your setup.

#### Execution example

```
python train.py --batch_size 512 --gpus 0 1 2 3 --nb_workers 12
```

You can resume training by setting `--resume_training` to True and defining the path to the model you want to resume training, `--load_model_path`.


### Evaluation

TBD

### Result examples

TBD



