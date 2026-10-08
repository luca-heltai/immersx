#!/bin/bash
set -e

export NODE_ADDR=127.0.0.1
echo "$NODE_ADDR slurmnode1" >> /etc/hosts

if [ -f /tmp/ssh_key.pub ]; then
  cp /tmp/ssh_key.pub /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

service munge start
service mariadb start
service slurmdbd start
service slurmctld start
slurmd -N slurmnode1
service ssh start

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec tail -f /dev/null
