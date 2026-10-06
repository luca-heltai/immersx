#!/bin/bash
set -e

mkdir -p "$ELECTRON_USERDATA"

# electron-store uses dealiix-storage.json, not config.json. Merge the tutorial
# target into that store without overwriting settings a returning user changed.
node <<'NODE'
const fs = require('fs')
const path = require('path')

const userData = process.env.ELECTRON_USERDATA
const storeFile = path.join(userData, 'dealiix-storage.json')
let state = {}
try {
  state = JSON.parse(fs.readFileSync(storeFile, 'utf8'))
} catch {}

const buildType = process.env.DEALIIX_BUILD_TYPE === 'Debug' ? 'Debug' : 'Release'
const buildSuffix = buildType === 'Debug' ? '_debug' : ''
const localDefaults = {
  coralBinaryPath: '', coralPluginPath: '', executablePath: '',
  parametersFileName: 'parameters.json', workingDirectory: '',
  mpiLauncher: { kind: 'mpirun' }, probes: {}
}
const remoteDefaults = {
  host: 'coral-ssh-slurm', port: 22, username: 'root',
  sshKeyPath: '/run/tutorial-ssh/id_ed25519',
  coralBinaryPath: `/usr/local/bin/coral-${buildType.toLowerCase()}`,
  coralPluginPath: `/opt/dealiix/immersx/${buildType}/lib/immersx/coral/libcoral_backend_immersx_2${buildSuffix}.so`,
  executablePath: '', parametersFileName: 'parameters.json',
  workingDirectory: '/app/shared-data',
  mpiLauncher: { kind: 'srun' }, probes: {}
}
const settings = state.settings && typeof state.settings === 'object' ? state.settings : {}
const execution = settings.execution && typeof settings.execution === 'object' ? settings.execution : {}
const local = execution.local && typeof execution.local === 'object' ? execution.local : {}
const remote = execution.remote && typeof execution.remote === 'object' ? execution.remote : {}
const tutorialPaths = !remote.coralBinaryPath ||
  remote.coralBinaryPath.startsWith('/usr/local/bin/coral-') ||
  remote.coralBinaryPath.startsWith('/opt/dealiix/')
const mergedRemote = {
  ...remoteDefaults, ...remote,
  mpiLauncher: { ...remoteDefaults.mpiLauncher, ...(remote.mpiLauncher || {}) }
}
if (tutorialPaths) {
  mergedRemote.coralBinaryPath = remoteDefaults.coralBinaryPath
  mergedRemote.coralPluginPath = remoteDefaults.coralPluginPath
}

state.settings = {
  ...settings,
  urlVisualizer: settings.urlVisualizer || 'http://coral-visualizer:8080',
  urlRemoteServer: settings.urlRemoteServer || 'http://localhost:8080',
  execution: {
    ...execution,
    local: { ...localDefaults, ...local, mpiLauncher: { ...localDefaults.mpiLauncher, ...(local.mpiLauncher || {}) } },
    remote: mergedRemote
  }
}
state.execution_selection = state.execution_selection || { location: 'remote', backendKind: 'coral' }
state.tutorial_build_type = buildType
fs.writeFileSync(storeFile, JSON.stringify(state, null, 2) + '\n')
NODE

Xvfb :99 -screen 0 1600x1000x24 -ac +extension GLX +render -noreset >/tmp/Xvfb.log 2>&1 &
fluxbox >/tmp/fluxbox.log 2>&1 &
x11vnc -display :99 -forever -shared -rfbport 5900 -nopw -quiet >/tmp/x11vnc.log 2>&1 &
cat > /usr/share/novnc/index.html <<'HTML'
<!doctype html>
<meta http-equiv="refresh" content="0; url=/vnc.html?autoconnect=1&resize=scale">
<a href="/vnc.html?autoconnect=1&resize=scale">Open dealiiX-platform</a>
HTML
/usr/share/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 6080 >/tmp/novnc.log 2>&1 &

cd /opt/dealiix-platform
exec node_modules/.bin/electron --no-sandbox .
