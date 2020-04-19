#!/usr/bin/env bash
. ~/.bashrc

PORT1='8555'
EXEC_STR='ssh -i /home/yonatanz/.ssh/id_rsa -L '$((${PORT1} + 10000))':127.0.0.1:'${PORT1}' morpheus@40.127.166.177'
gnome-terminal --tab -- ${EXEC_STR}

EXEC_STR="ssh -i /home/yonatanz/.ssh/id_rsa morpheus@40.127.166.177 bash -l '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/scripts/jupyter.sh' "${PORT1}
gnome-terminal --tab -- ${EXEC_STR}

echo 'jupyter notebook port is '${PORT1}

PORT2='8475'
EXEC_STR='ssh -i /home/yonatanz/.ssh/id_rsa -L '$((${PORT2} + 10000))':127.0.0.1:'${PORT2}' morpheus@40.127.166.177'
gnome-terminal --tab -- ${EXEC_STR}

EXEC_STR="ssh -i /home/yonatanz/.ssh/id_rsa morpheus@40.127.166.177 bash -l '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/scripts/tensorboard.sh' "${PORT2}
gnome-terminal --tab -- ${EXEC_STR}

echo 'tensorboard port is '${PORT2}
