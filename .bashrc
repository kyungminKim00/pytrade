#!/bin/bash

res=$(service ssh status | head -n 1)

if [ "$res" == " * sshd is not running" ];
then
        service ssh start
else
        echo " * sshd is running"
fi
