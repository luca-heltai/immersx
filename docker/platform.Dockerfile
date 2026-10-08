FROM node:22-bookworm AS build

WORKDIR /opt/dealiix-platform
COPY --from=platform / /opt/dealiix-platform/
RUN npm ci && npm run build

FROM node:22-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV ELECTRON_USERDATA=/var/lib/dealiix-platform

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb x11vnc novnc fluxbox \
    libgtk-3-0 libnss3 libgbm1 libasound2 libxss1 libxtst6 \
    libxshmfence1 libdrm2 libnotify4 libsecret-1-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/dealiix-platform /opt/dealiix-platform
COPY docker/platform-entrypoint.sh /usr/local/bin/platform-entrypoint.sh
RUN chmod +x /usr/local/bin/platform-entrypoint.sh && mkdir -p /var/lib/dealiix-platform

EXPOSE 6080
ENTRYPOINT ["/usr/local/bin/platform-entrypoint.sh"]
