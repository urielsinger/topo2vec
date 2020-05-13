import os
import sys

from topo2vec.constants import BASE_LOCATION

print(sys.executable)

###############################################
# run the program using nohup - in background #
###############################################

filname = 'autoencoder_experiment_hyperparams'
nohup_path = BASE_LOCATION + 'topo2vec/data/nohup_results/'
os.system("nohup sh -c '" +
          sys.executable + f" {filname}.py >{nohup_path}res_{filname}.txt" +
          "' &")