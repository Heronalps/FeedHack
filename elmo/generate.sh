python bin/dump_weights.py --save_dir ./checkpoint_2 --outfile ./checkpoint_2/weights.hdf5

allennlp elmo sentences.txt ./checkpoint_2/weights.hdf5 --average

python generate.py
