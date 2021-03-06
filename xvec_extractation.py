#x-vectors from the .ark files for single speakers 

import numpy as np
import kaldiio
import os

for filename in os.listdir('espnet/egs/libritts/tts1/exp/xvector_nnet_1a/xvectors_train_clean_100'):
  if filename.endswith('.ark'):
    f = kaldiio.load_ark(os.path.join('espnet/egs/libritts/tts1/exp/xvector_nnet_1a/xvectors_train_clean_100', filename)) # open in readonly mode
    for key, numpy_array in f:
      print(filename,key)
      path = 'xvector/'+key+'.txt'
      print(path)
      np.savetxt(path, numpy_array)
