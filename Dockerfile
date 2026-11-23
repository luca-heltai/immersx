# syntax=docker/dockerfile:1
ARG CORAL_IMAGE=ghcr.io/2listic/coral:main
ARG METRIC_FLOW_X_IMAGE=ghcr.io/luca-heltai/metric-flow-x:main

FROM ${METRIC_FLOW_X_IMAGE} AS metric-flow-x-sdk

FROM ${CORAL_IMAGE} AS build
USER root
COPY --from=metric-flow-x-sdk /opt/metric-flow-x /opt/metric-flow-x
ARG METRIC_FLOW_X_IMAGE
RUN if ! test -x /opt/metric-flow-x/bin/blood_flow_debug || \
       ! grep -Eq '^#define METRIC_FLOW_X_WITH_TRILINOS +1' /opt/metric-flow-x/include/metric_flow_x/config.h; then \
      printf '%s must contain the Debug and Release Trilinos SDK.\n' "$METRIC_FLOW_X_IMAGE" >&2; \
      exit 1; \
    fi
WORKDIR /src
COPY CMakeLists.txt LICENSE.md README.md ./
COPY apps/ apps/
COPY include/ include/
COPY source/ source/
COPY cmake/ cmake/
COPY coral/ coral/
COPY data/ data/
# The legacy reduced_poisson entry point includes this header at compile time.
COPY tests/tests.h tests/tests.h
ARG BUILD_JOBS=2
RUN set -eu; for config in Release Debug; do \
      prefix=/opt/immersx; \
      if [ "$config" = Debug ]; then prefix=/opt/immersx-debug; fi; \
      cmake -S . -B "/build/$config" -GNinja -DCMAKE_BUILD_TYPE="$config" \
        -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_INSTALL_RPATH="$prefix/lib" \
        -Dcoral_DIR="/opt/dealiix/coral/$config/lib/cmake/coral" \
        -DCMAKE_PREFIX_PATH=/opt/metric-flow-x \
        -DIMMERSX_BUILD_CORAL_PLUGINS=ON -DIMMERSX_ENABLE_CORAL_GRAPH_TESTS=OFF \
        -DENABLE_GOOGLE_TESTING=OFF -DENABLE_DEAL_II_APP_TESTING=OFF \
        -DENABLE_COUPLED_PROBLEMS=OFF -DIMMERSX_INSTALL_TESTS=OFF \
        -DIMMERSX_USE_CCACHE=OFF > "/tmp/immersx-$config-configure.log" \
        || { cat "/tmp/immersx-$config-configure.log"; exit 1; }; \
      grep -F "MetricFlowX support enabled" "/tmp/immersx-$config-configure.log"; \
      grep -F "ImmersX Coral plugins enabled" "/tmp/immersx-$config-configure.log"; \
      cmake --build "/build/$config" -j "$BUILD_JOBS"; \
      cmake --install "/build/$config"; \
    done \
    && rm /opt/immersx/bin/coupled_elasticity /opt/immersx/bin/pseudocoupling1D

FROM ${CORAL_IMAGE}
USER root
COPY --from=build /opt/immersx /opt/immersx
COPY --from=build /opt/immersx-debug /opt/immersx-debug
COPY --from=metric-flow-x-sdk /opt/metric-flow-x /opt/metric-flow-x
COPY --chmod=755 scripts/container_entrypoint.sh /opt/immersx/bin/immersx
ARG CORAL_IMAGE
ARG METRIC_FLOW_X_IMAGE
ARG VCS_REF=unknown
ARG VERSION=master
LABEL org.opencontainers.image.title="ImmersX" \
      org.opencontainers.image.source="https://github.com/luca-heltai/immersx" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      io.github.luca-heltai.immersx.coral-image="${CORAL_IMAGE}" \
      io.github.luca-heltai.immersx.metric-flow-x-image="${METRIC_FLOW_X_IMAGE}"
ENV PATH="/opt/immersx/bin:/opt/metric-flow-x/bin:${PATH}" \
    CMAKE_PREFIX_PATH="/opt/immersx:/opt/metric-flow-x:/opt/dealiix/coral/Release" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN printf '%s\n' \
      '#!/bin/sh' \
      'export LD_LIBRARY_PATH=/opt/dealiix/coral/Release/lib/Release:/opt/metric-flow-x/lib:/opt/immersx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' \
      'exec /opt/dealiix/coral/Release/bin/Release/coral "$@"' \
      > /usr/local/bin/coral-release \
    && printf '%s\n' \
      '#!/bin/sh' \
      'export LD_LIBRARY_PATH=/opt/dealiix/coral/Debug/lib/Debug:/opt/metric-flow-x/lib:/opt/immersx-debug/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' \
      'exec /opt/dealiix/coral/Debug/bin/Debug/coral "$@"' \
      > /usr/local/bin/coral-debug \
    && chmod +x /usr/local/bin/coral-release /usr/local/bin/coral-debug \
    && ldconfig && mkdir -p /work && chmod 1777 /work

WORKDIR /work
USER 1000:1000
ENTRYPOINT ["/opt/immersx/bin/immersx"]
CMD []
