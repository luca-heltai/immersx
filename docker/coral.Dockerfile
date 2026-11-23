# syntax=docker/dockerfile:1
ARG CORAL_IMAGE=ghcr.io/2listic/coral:main
ARG METRIC_FLOW_X_IMAGE=ghcr.io/luca-heltai/metric-flow-x:main

FROM ${METRIC_FLOW_X_IMAGE} AS metric-flow-x-sdk

FROM ${CORAL_IMAGE} AS build
USER root
COPY --from=metric-flow-x-sdk /opt/metric-flow-x /opt/metric-flow-x
# The installed SDK must match the Trilinos API used by ImmersX.
RUN test -x /opt/metric-flow-x/bin/blood_flow_debug \
    && grep -Eq '^#define METRIC_FLOW_X_WITH_TRILINOS +1' /opt/metric-flow-x/include/metric_flow_x/config.h
COPY . /src/immersx
WORKDIR /src/immersx
ARG BUILD_JOBS=2
RUN set -eu; for config in Debug Release; do \
      cmake -S . -B "/build/$config" -GNinja \
        -DCMAKE_BUILD_TYPE="$config" -DCMAKE_INSTALL_PREFIX="/opt/dealiix/immersx/$config" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -Dcoral_DIR="/opt/dealiix/coral/$config/lib/cmake/coral" \
        -DCMAKE_PREFIX_PATH=/opt/metric-flow-x \
        -DIMMERSX_BUILD_CORAL_PLUGINS=ON -DIMMERSX_ENABLE_CORAL_GRAPH_TESTS=OFF \
        -DIMMERSX_INSTALL_TESTS=OFF -DENABLE_COUPLED_PROBLEMS=OFF \
        -DENABLE_GOOGLE_TESTING=OFF -DENABLE_DEAL_II_APP_TESTING=OFF \
        > "/tmp/immersx-$config-configure.log" \
        || { cat "/tmp/immersx-$config-configure.log"; exit 1; }; \
      grep -F "MetricFlowX support enabled" "/tmp/immersx-$config-configure.log"; \
      grep -F "ImmersX Coral plugins enabled" "/tmp/immersx-$config-configure.log"; \
      cmake --build "/build/$config" -j "$BUILD_JOBS"; \
      cmake --install "/build/$config"; \
    done

FROM ${CORAL_IMAGE}
USER root
COPY --from=build /opt/dealiix/immersx /opt/dealiix/immersx
COPY --from=metric-flow-x-sdk /opt/metric-flow-x /opt/metric-flow-x
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/luca-heltai/immersx" \
      org.opencontainers.image.revision="${VCS_REF}"

RUN apt-get -o Acquire::Retries=3 update && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
    sudo munge slurmd slurm-client slurmctld slurmdbd mariadb-server \
    openssh-server python3 python3-pip python3-venv less vim \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_* /etc/munge/munge.key

RUN mkdir -p /run/sshd /root/.ssh /var/log/slurm /var/spool/slurmctld \
    /var/spool/slurmd.slurmnode1
RUN printf '%s\n' 'PermitRootLogin prohibit-password' 'PasswordAuthentication no' \
    > /etc/ssh/sshd_config.d/00-tutorial.conf

COPY docker/slurm.conf /etc/slurm/slurm.conf
COPY docker/slurmdbd.conf /etc/slurm/slurmdbd.conf
COPY docker/cgroup.conf /etc/slurm/cgroup.conf
RUN chmod 600 /etc/slurm/slurmdbd.conf

RUN printf '%s\n' \
      '#!/bin/sh' \
      'export LD_LIBRARY_PATH=/opt/dealiix/coral/Debug/lib/Debug:/opt/metric-flow-x/lib:/opt/dealiix/immersx/Debug/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' \
      'exec /opt/dealiix/coral/Debug/bin/Debug/coral "$@"' \
      > /usr/local/bin/coral-debug \
    && printf '%s\n' \
      '#!/bin/sh' \
      'export LD_LIBRARY_PATH=/opt/dealiix/coral/Release/lib/Release:/opt/metric-flow-x/lib:/opt/dealiix/immersx/Release/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' \
      'exec /opt/dealiix/coral/Release/bin/Release/coral "$@"' \
      > /usr/local/bin/coral-release \
    && chmod +x /usr/local/bin/coral-debug /usr/local/bin/coral-release \
    && ln -sf /usr/local/bin/coral-release /usr/local/bin/coral
ENV PATH="/opt/metric-flow-x/bin:${PATH}"
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

COPY docker/coral-entrypoint.sh /usr/local/bin/coral-entrypoint.sh
RUN chmod +x /usr/local/bin/coral-entrypoint.sh

WORKDIR /app/shared-data
RUN mkdir -p /app/shared-data
EXPOSE 22 6817 6818
ENTRYPOINT ["/usr/local/bin/coral-entrypoint.sh"]
