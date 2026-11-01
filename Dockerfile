# syntax=docker/dockerfile:1
ARG DEAL_II_IMAGE=dealii/dealii:v9.7.1-noble@sha256:28a40c95e76879c717a96ea3f5c87cc00ec1d7a4ce9d250ab936fb723eab6995
FROM ${DEAL_II_IMAGE} AS build
USER root
WORKDIR /src
COPY CMakeLists.txt LICENSE.md README.md ./
COPY apps/ apps/
COPY include/ include/
COPY source/ source/
COPY cmake/ cmake/
COPY data/ data/
# The legacy reduced_poisson entry point includes this header at compile time.
COPY tests/tests.h tests/tests.h
ARG BUILD_JOBS=2
RUN cmake -S . -B /build -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/immersx -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_INSTALL_RPATH=/opt/immersx/lib \
      -DENABLE_GOOGLE_TESTING=OFF -DENABLE_DEAL_II_APP_TESTING=OFF \
      -DENABLE_COUPLED_PROBLEMS=OFF -DIMMERSX_BUILD_CORAL_PLUGINS=OFF \
      -DIMMERSX_INSTALL_TESTS=OFF -DIMMERSX_USE_CCACHE=OFF \
    && cmake --build /build -j "${BUILD_JOBS}" \
    && cmake --install /build \
    && rm /opt/immersx/bin/coupled_elasticity /opt/immersx/bin/pseudocoupling1D

# Keep the matching deal.II shared dependencies and SDK, as in MetricFlowX.
FROM ${DEAL_II_IMAGE}
USER root
COPY --from=build /opt/immersx /opt/immersx
COPY --chmod=755 scripts/container_entrypoint.sh /opt/immersx/bin/immersx
ARG VCS_REF=unknown
ARG VERSION=master
LABEL org.opencontainers.image.title="ImmersX" \
      org.opencontainers.image.source="https://github.com/luca-heltai/immersx" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"
ENV PATH="/opt/immersx/bin:${PATH}" \
    CMAKE_PREFIX_PATH="/opt/immersx" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
RUN ldconfig && mkdir /work && chmod 1777 /work
WORKDIR /work
USER 1000:1000
ENTRYPOINT ["/opt/immersx/bin/immersx"]
CMD []
