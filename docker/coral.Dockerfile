# syntax=docker/dockerfile:1
ARG DEAL_II_IMAGE=dealii/dealii:v9.7.1-noble@sha256:28a40c95e76879c717a96ea3f5c87cc00ec1d7a4ce9d250ab936fb723eab6995
ARG CORAL_SDK=coral-source
ARG METRIC_FLOW_X_SDK=metric-flow-x-source

FROM ${DEAL_II_IMAGE} AS coral-source
USER root
COPY --from=coral / /src/coral
WORKDIR /src/coral
ARG BUILD_JOBS=2
RUN set -eu; for config in Debug Release; do \
      cmake -S . -B "/build/$config" -GNinja \
        -DCMAKE_BUILD_TYPE="$config" -DCMAKE_INSTALL_PREFIX="/opt/dealiix/coral/$config" \
        -DCMAKE_INSTALL_LIBDIR=lib -DCORAL_BUILD_BACKEND_DEALII=OFF \
        -DCORAL_BUILD_SHARED_CORE=ON -DCORAL_BUILD_TESTS=OFF -DCORAL_INSTALL=ON; \
      cmake --build "/build/$config" -j "$BUILD_JOBS"; \
      cmake --install "/build/$config"; \
    done

FROM ${DEAL_II_IMAGE} AS metric-flow-x-source
USER root
COPY --from=metric_flow_x / /src/metric-flow-x
WORKDIR /src/metric-flow-x
ARG BUILD_JOBS=2
RUN cmake -S . -B /build -GNinja -DCMAKE_BUILD_TYPE=DebugRelease \
      -DCMAKE_INSTALL_PREFIX=/opt/metric-flow-x -DMETRIC_FLOW_X_BACKEND=Trilinos \
    && cmake --build /build --target blood_flow blood_flow_debug -j "$BUILD_JOBS" \
    && cmake --install /build

# These stages can also select GHCR image digests with the same SDK layout.
FROM ${CORAL_SDK} AS coral-sdk
FROM ${METRIC_FLOW_X_SDK} AS metric-flow-x-sdk

FROM ${DEAL_II_IMAGE} AS build
USER root
COPY --from=coral-sdk /opt/dealiix/coral /opt/dealiix/coral
COPY --from=metric-flow-x-sdk /opt/metric-flow-x /opt/metric-flow-x
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

FROM ${DEAL_II_IMAGE}
USER root
COPY --from=build /opt/dealiix /opt/dealiix
COPY --from=build /opt/metric-flow-x /opt/metric-flow-x
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
