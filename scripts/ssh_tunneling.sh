#!/usr/bin/env bash
PORT1='8576'
EXEC_STR='ssh -i /home/yonatanz/.ssh/id_rsa -L '$((${PORT1} + 10000))':127.0.0.1:'${PORT1}' morpheus@40.127.166.177'
gnome-terminal --tab -- ${EXEC_STR}


PORT2='8444'
EXEC_STR='ssh -i /home/yonatanz/.ssh/id_rsa -L '$((${PORT2} + 10000))':127.0.0.1:'${PORT2}' morpheus@40.127.166.177'
gnome-terminal --tab -- ${EXEC_STR}