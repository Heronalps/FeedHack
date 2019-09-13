import h5py
import pandas as pd
from collections import OrderedDict 
import numpy as np

elmo = h5py.File("./checkpoint_2/weights.hdf5", 'r')   
stack = pd.read_csv("./stack_labeled_feedback.txt", sep='\t')

output_dict = OrderedDict()
for key in range(len(elmo) - 1):
    #import pdb; pdb.set_trace();
    vector_list = elmo.get(str(key)).value
    output_dict[int(key)] = [stack['ConformedFeedbackId'][int(key)], vector_list]

df = pd.DataFrame.from_dict(data=output_dict, orient='index')
df.rename(columns={0:'ConformedFeedbackId', 1:'Embeddings'}, inplace=True)

df['Avg_Embeddings'] = df['Embeddings'].apply(lambda vector: np.average(vector, axis=0))
df_avg = pd.concat([df['ConformedFeedbackId'], df['Avg_Embeddings']], axis=1)

df_split = pd.DataFrame(df_avg['Avg_Embeddings'].values.tolist())

df_output = pd.concat([df_avg['ConformedFeedbackId'], df_split], axis=1)


df_output.to_csv("./embed.csv", header=True, index=False)

