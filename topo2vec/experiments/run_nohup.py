import os
import sys
print(sys.executable)
filname = 'autoencoder_experiment_hyperparams'
nohup_path = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/nohup_results/'
os.system("nohup sh -c '" +
          sys.executable + f" {filname}.py >{nohup_path}res_{filname}.txt" +
          "' &")