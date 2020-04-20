#!/usr/bin/env bash
. ~/.bashrc

PORT1='8119'
USER_IP='root@159.122.160.134'

EXEC_STR='ssh -i /home/yonatanz/.ssh/id_rsa -L '$((${PORT1}))':127.0.0.1:'${PORT1}' '${USER_IP}
gnome-terminal --tab -- ${EXEC_STR}

EXEC_STR="ssh -i /home/yonatanz/.ssh/id_rsa root@159.122.160.134 bash -li '/home/topo2vec_kavitzky/topo2vec/scripts/jupyter.sh' "${PORT1}
gnome-terminal --tab -- ${EXEC_STR}

echo 'jupyter notebook port is '${PORT1}

PORT2='8229'
EXEC_STR='ssh -i /home/yonatanz/.ssh/id_rsa -L '$((${PORT2}))':127.0.0.1:'${PORT2}' '${USER_IP}
gnome-terminal --tab -- ${EXEC_STR}

EXEC_STR="ssh -i /home/yonatanz/.ssh/id_rsa root@159.122.160.134 bash -li '/home/topo2vec_kavitzky/topo2vec/scripts/tensorboard.sh' "${PORT2}
gnome-terminal --tab -- ${EXEC_STR}

echo 'tensorboard port is '${PORT2}
