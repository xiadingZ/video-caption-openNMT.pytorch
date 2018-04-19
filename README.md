# Video caption usage


0. preparocess data

`python prepro_feats.py --output_dir data/feats/resnet152_origin --model resnet152 --n_frame_steps 40  --gpu 4,5`

then, use [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to
 extract features from video. Then mean pool to get a 2048 dim feature for each video, save each feature as 
 `{video id}_c3d.npy`, such as `video0_c3d.npy`, put them under `data/feats/resnet152_origin`.
 
 Then write `src-train.txt, tgt-train.txt, src-val.txt, tgt-val.txt, src-test.txt, tgt-test.txt`, put them under 
 `data\src_train.txt` has video id line by line, such as `video0 \n video1`, `tgt_train.txt` has captions line by line, 
 each line is a caption corresponding to video id in `src_train.txt`. Other files' format is the same as these two.


1. train

```bash
python train.py -model_type video -data data/msrvtt/video -save_model data/save/model -gpuid 4 -batch_size 180 -max_grad_norm 20 -dim_vid 4096 -rnn_size 1024 -optim adam -learning_rate 0.001 -epochs 250  -dropout 0.5 -global_attention mlp -encoder_type brnn
```

2. translate

```bash
python translate.py -data_type video -model data/nmt/model_acc_41.25_ppl_35.38_e12.pt -src_dir data/feats/resnet152_origin -src data/src-test.txt -output pred.txt -gpu 1
```

3. eval

```bash
python eval.py  -video_ids data/src-test.txt -pred pred.txt
```

# OpenNMT-py: Open-Source Neural Machine Translation

[![Build Status](https://travis-ci.org/OpenNMT/OpenNMT-py.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-py)

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system. It is designed to be research friendly to try out new ideas in translation, summary, image-to-text, morphology, and many other domains.

Codebase is relatively stable, but PyTorch is still evolving. We currently recommend forking if you need to have stable code.

OpenNMT-py is run as a collaborative open-source project. It is maintained by [Sasha Rush](http://github.com/srush) (Cambridge, MA), [Ben Peters](http://github.com/bpopeters) (Saarbrücken), and [Jianyu Zhan](http://github.com/jianyuzhan) (Shenzhen). The original code was written by [Adam Lerer](http://github.com/adamlerer) (NYC). 
We love contributions. Please consult the Issues page for any [Contributions Welcome](https://github.com/OpenNMT/OpenNMT-py/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) tagged post. 

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>


Table of Contents
=================
  * [Full Documentation](http://opennmt.net/OpenNMT-py/)
  * [Requirements](#requirements)
  * [Features](#features)
  * [Quickstart](#quickstart)
  * [Citation](#citation)
 
## Requirements

```bash
pip install -r requirements.txt
```


## Features

The following OpenNMT features are implemented:

- [data preprocessing](http://opennmt.net/OpenNMT-py/options/preprocess.html)
- [Inference (translation) with batching and beam search](http://opennmt.net/OpenNMT-py/options/translate.html)
- [Multiple source and target RNN (lstm/gru) types and attention (dotprod/mlp) types](http://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [TensorBoard/Crayon logging](http://opennmt.net/OpenNMT-py/options/train.html#logging)
- [Source word features](http://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [Pretrained Embeddings](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Copy and Coverage Attention](http://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Image-to-text processing](http://opennmt.net/OpenNMT-py/im2text.html)
- [Speech-to-text processing](http://opennmt.net/OpenNMT-py/speech2text.html)
- ["Attention is all you need"](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- Inference time loss functions.

Beta Features (committed):
- multi-GPU
- Structured attention
- [Conv2Conv convolution model]
- SRU "RNNs faster than CNN" paper

## Quickstart

[Full Documentation](http://opennmt.net/OpenNMT-py/)


### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
python train.py -data data/demo -save_model demo-model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder. You
can also add `-gpuid 1` to use (say) GPU 1.

### Step 3: Translate

```bash
python translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

!!! note "Note"
    The predictions are going to be quite terrible, as the demo dataset is small. Try running on some larger datasets! For example you can download millions of parallel sentences for [translation](http://www.statmt.org/wmt16/translation-task.html) or [summarization](https://github.com/harvardnlp/sent-summary).

## Pretrained embeddings (e.g. GloVe)

Go to tutorial: [How to use GloVe pre-trained embeddings in OpenNMT-py](http://forum.opennmt.net/t/how-to-use-glove-pre-trained-embeddings-in-opennmt-py/1011)

## Pretrained Models

The following pretrained models can be downloaded and used with translate.py.

http://opennmt.net/Models-py/



## Citation

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
