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
RUN for cfg in Debug Release; do \
      cmake -S . -B /tmp/coral-build-$cfg \
        -DCMAKE_BUILD_TYPE=$cfg \
        -DCMAKE_INSTALL_PREFIX=/opt/dealiix/coral/$cfg \
        -DCORAL_BUILD_BACKEND_DEALII=OFF \
        -DCORAL_BUILD_SHARED_CORE=ON \
        -DCORAL_BUILD_TESTS=OFF \
        -DCORAL_INSTALL=ON; \
      cmake --build /tmp/coral-build-$cfg --parallel 2; \
      cmake --install /tmp/coral-build-$cfg; \
    done

COPY --from=metric_flow_x / /src/metric-flow-x
WORKDIR /src/metric-flow-x
RUN for cfg in Debug Release; do \
      cmake -S . -B /tmp/metric-flow-x-build-$cfg \
        -DCMAKE_BUILD_TYPE=$cfg \
        -DCMAKE_INSTALL_PREFIX=/opt/dealiix/metric-flow-x/$cfg; \
      cmake --build /tmp/metric-flow-x-build-$cfg --parallel 2 \
        --target metric_flow_x blood_flow; \
      cmake --install /tmp/metric-flow-x-build-$cfg; \
    done

COPY . /src/immersx
WORKDIR /src/immersx
RUN cmake -S . -B /tmp/immersx-build-Debug \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=/opt/dealiix/immersx/Debug \
      -Dcoral_DIR=/opt/dealiix/coral/Debug/lib/cmake/coral \
      -DMetricFlowX_DIR=/opt/dealiix/metric-flow-x/Debug/lib/cmake/MetricFlowX \
      -DIMMERSX_BUILD_CORAL_PLUGINS=ON \
      -DIMMERSX_ENABLE_CORAL_GRAPH_TESTS=OFF \
      -DIMMERSX_INSTALL_TESTS=OFF \
      -DENABLE_COUPLED_PROBLEMS=OFF \
      -DENABLE_GOOGLE_TESTING=OFF \
      -DENABLE_DEAL_II_APP_TESTING=OFF \
    && cmake --build /tmp/immersx-build-Debug --parallel 2 \
    && cmake --install /tmp/immersx-build-Debug

RUN cmake -S . -B /tmp/immersx-build-Release \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/dealiix/immersx/Release \
      -Dcoral_DIR=/opt/dealiix/coral/Release/lib/cmake/coral \
      -DMetricFlowX_DIR=/opt/dealiix/metric-flow-x/Release/lib/cmake/MetricFlowX \
      -DIMMERSX_BUILD_CORAL_PLUGINS=ON \
      -DIMMERSX_ENABLE_CORAL_GRAPH_TESTS=OFF \
      -DIMMERSX_INSTALL_TESTS=OFF \
      -DENABLE_COUPLED_PROBLEMS=OFF \
      -DENABLE_GOOGLE_TESTING=OFF \
      -DENABLE_DEAL_II_APP_TESTING=OFF \
    && cmake --build /tmp/immersx-build-Release --parallel 2 \
    && cmake --install /tmp/immersx-build-Release

RUN printf '%s\n' \
      '#!/bin/sh' \
      'export LD_LIBRARY_PATH=/opt/dealiix/coral/Debug/lib:/opt/dealiix/metric-flow-x/Debug/lib:/opt/dealiix/immersx/Debug/lib:${LD_LIBRARY_PATH}' \
      'exec /opt/dealiix/coral/Debug/bin/Debug/coral "$@"' \
      > /usr/local/bin/coral-debug \
    && printf '%s\n' \
      '#!/bin/sh' \
      'export LD_LIBRARY_PATH=/opt/dealiix/coral/Release/lib:/opt/dealiix/metric-flow-x/Release/lib:/opt/dealiix/immersx/Release/lib:${LD_LIBRARY_PATH}' \
      'exec /opt/dealiix/coral/Release/bin/Release/coral "$@"' \
      > /usr/local/bin/coral-release \
    && chmod +x /usr/local/bin/coral-debug /usr/local/bin/coral-release \
    && ln -sf /usr/local/bin/coral-release /usr/local/bin/coral
ENV LD_LIBRARY_PATH=/opt/dealiix/coral/Release/lib:/opt/dealiix/coral/Debug/lib:/opt/dealiix/metric-flow-x/Release/lib:/opt/dealiix/metric-flow-x/Debug/lib:/opt/dealiix/immersx/Release/lib:/opt/dealiix/immersx/Debug/lib
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

COPY docker/coral-entrypoint.sh /usr/local/bin/coral-entrypoint.sh
RUN chmod +x /usr/local/bin/coral-entrypoint.sh

WORKDIR /app/shared-data
RUN mkdir -p /app/shared-data
EXPOSE 22 6817 6818
ENTRYPOINT ["/usr/local/bin/coral-entrypoint.sh"]
