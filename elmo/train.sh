#! /bin/bash
git clone https://github.com/allenai/bilm-tf.git
cd bilm-tf
mkdir files
cd files

curl https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/vocab-2016-09-10.txt --output vocab.txt
curl https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/checkpoint --output checkpoint
curl https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/options.json --output options.json

curl https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/model.ckpt-935588.data-00000-of-00001 --output model.ckpt-935588.data-00000-of-00001
curl https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/model.ckpt-935588.index --output model.ckpt-935588.index
curl https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/model.ckpt-935588.meta --output model.ckpt-935588.meta

cd ..

export CUDA_VISIBLE_DEVICES=0,1,2,3
python bin/restart.py \
     --train_prefix='./uif_pretrain_100k_No_Space.txt' \
     --vocab_file ./files/vocab.txt \
     --save_dir ./files