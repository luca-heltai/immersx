#!/bin/bash
set -e

mkdir -p "$ELECTRON_USERDATA"

if [ ! -f "$ELECTRON_USERDATA/config.json" ]; then
  cat > "$ELECTRON_USERDATA/config.json" <<'JSON'
{"settings":{"urlVisualizer":"http://coral-visualizer:8080","urlRemoteServer":"http://localhost:8080","execution":{"local":{"coralBinaryPath":"","coralPluginPath":"","executablePath":"","parametersFileName":"parameters.json","workingDirectory":"","mpiLauncher":{"kind":"mpirun"},"probes":{}},"remote":{"host":"coral-ssh-slurm","port":22,"username":"root","sshKeyPath":"/run/tutorial-ssh/id_ed25519","coralBinaryPath":"/opt/dealiix/coral/bin/Release/coral","coralPluginPath":"/opt/dealiix/immersx/lib/immersx/coral/libcoral_backend_immersx_2.so","executablePath":"","parametersFileName":"parameters.json","workingDirectory":"/app/shared-data","mpiLauncher":{"kind":"srun"},"probes":{}}}},"execution_selection":{"location":"remote","backendKind":"coral"}}
JSON
fi

Xvfb :99 -screen 0 1600x1000x24 -ac +extension GLX +render -noreset >/tmp/Xvfb.log 2>&1 &
fluxbox >/tmp/fluxbox.log 2>&1 &
x11vnc -display :99 -forever -shared -rfbport 5900 -nopw -quiet >/tmp/x11vnc.log 2>&1 &
/usr/share/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 6080 >/tmp/novnc.log 2>&1 &

cd /opt/dealiix-platform
exec node_modules/.bin/electron --no-sandbox .
