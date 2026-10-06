FROM dealii/dealii:v9.7.1-noble

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo munge slurmd slurm-client slurmctld slurmdbd mariadb-server \
    openssh-server python3 python3-pip python3-venv less vim \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /run/sshd /root/.ssh /var/log/slurm /var/spool/slurmctld \
    /var/spool/slurmd.slurmnode1
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config

COPY docker/slurm.conf /etc/slurm/slurm.conf
COPY docker/slurmdbd.conf /etc/slurm/slurmdbd.conf
COPY docker/cgroup.conf /etc/slurm/cgroup.conf
RUN chmod 600 /etc/slurm/slurmdbd.conf

COPY --from=coral / /src/coral
WORKDIR /src/coral
RUN cmake -S . -B /tmp/coral-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/dealiix/coral \
      -DCORAL_BUILD_BACKEND_DEALII=OFF \
      -DCORAL_BUILD_SHARED_CORE=ON \
      -DCORAL_BUILD_TESTS=OFF \
      -DCORAL_INSTALL=ON \
    && cmake --build /tmp/coral-build --parallel 2 \
    && cmake --install /tmp/coral-build

COPY . /src/immersx
WORKDIR /src/immersx
RUN cmake -S . -B /tmp/immersx-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/dealiix/immersx \
      -Dcoral_DIR=/opt/dealiix/coral/lib/cmake/coral \
      -DIMMERSX_BUILD_CORAL_PLUGINS=ON \
      -DIMMERSX_ENABLE_CORAL_GRAPH_TESTS=OFF \
      -DENABLE_COUPLED_PROBLEMS=OFF \
      -DENABLE_GOOGLE_TESTING=OFF \
      -DENABLE_DEAL_II_APP_TESTING=OFF \
    && cmake --build /tmp/immersx-build --parallel 2 \
      --target coral_backend_immersx_1 coral_backend_immersx_2 coral_backend_immersx_3 \
    && cmake --install /tmp/immersx-build

RUN ln -s /opt/dealiix/coral/bin/Release/coral /usr/local/bin/coral
ENV LD_LIBRARY_PATH=/opt/dealiix/coral/lib/Release:/opt/dealiix/immersx/lib
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

COPY docker/coral-entrypoint.sh /usr/local/bin/coral-entrypoint.sh
RUN chmod +x /usr/local/bin/coral-entrypoint.sh

WORKDIR /app/shared-data
RUN mkdir -p /app/shared-data
EXPOSE 22 6817 6818
ENTRYPOINT ["/usr/local/bin/coral-entrypoint.sh"]
