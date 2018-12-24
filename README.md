# video interpolation

### Basic intro

This is my personal implementation of video interpolation method descried in my graduate thesis. It is a two-step method:flow+synthesis, but with nerual networks. Thanks to Simon Niklaus's fantastic papers about video interpolation and his pytorch implementation about pwc-net. 

### Training
I train the whole network with vimeo interp. dataset. You can find it here[http://toflow.csail.mit.edu/].
Download the training set and start training. You may need to make some changes in train.py.
```
python3 train.py 
```

### Testing
```
python3 test.py
```

### Run a demo
