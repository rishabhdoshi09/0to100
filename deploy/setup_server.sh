#!/usr/bin/env bash
# Install/update QuantTerm through the one canonical installed-host path.
#
# The systemd user service owns product.host_entrypoint, which owns exactly one
# product.host_supervisor and its five children. Historical split UI/autonomy
# services are removed before installation so there is one mutation owner.
set -euo pipefail

if [[ "$(id -u)" == "0" ]]; then
  echo "Run setup_server.sh as the intended QuantTerm login user, not root; sudo is used only for OS packages/service cleanup." >&2
  exit 1
fi

REPO_URL="${QT_REPO_URL:-https://github.com/rishabhdoshi09/0to100.git}"
SCRIPT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -d "$SCRIPT_ROOT/.git" ]]; then
  APP_DIR="${QT_DIR:-$SCRIPT_ROOT}"
else
  APP_DIR="${QT_DIR:-$HOME/0to100}"
fi
BRANCH="${QT_BRANCH:-claude/build-ai-trading-system-miHHd}"
RUNTIME_ROOT="${QT_RUNTIME_ROOT:-$HOME/.local/state/quantterm/runtime}"

if [[ -d "$APP_DIR/.git" ]]; then
  if [[ -n "${QT_BRANCH:-}" ]]; then
    git -C "$APP_DIR" fetch origin "$BRANCH"
    git -C "$APP_DIR" checkout "$BRANCH"
    git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
  fi
else
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

SHA="$(git -C "$APP_DIR" rev-parse HEAD)"
echo "== QuantTerm canonical host install: $SHA at $APP_DIR =="
sudo timedatectl set-timezone Asia/Kolkata 2>/dev/null || true
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip build-essential cmake curl nodejs npm

cd "$APP_DIR"
[ -d venv ] || python3 -m venv venv
PYTHON_BIN="${QT_PYTHON:-$APP_DIR/venv/bin/python}"
"$PYTHON_BIN" -m pip install --upgrade pip wheel
"$PYTHON_BIN" -m pip install -r requirements.txt
[ -f .env ] || { cp .env.example .env 2>/dev/null || touch .env; }
chmod 600 .env

# Retire all historical system-level split services. The canonical installer
# below creates one user-level quantterm.service with bounded restart semantics.
for legacy in quantterm-autonomy.service quantterm-ui.service quantterm.service; do
  sudo systemctl disable --now "$legacy" 2>/dev/null || true
  sudo rm -f "/etc/systemd/system/$legacy"
done
sudo systemctl daemon-reload

export PYTHON="$PYTHON_BIN"
"$APP_DIR/scripts/install_quantterm_host.sh" \
  --runtime-root "$RUNTIME_ROOT" \
  --env-file "$APP_DIR/.env" \
  --manager systemd

"$APP_DIR/scripts/quantterm_status.sh" \
  --runtime-root "$RUNTIME_ROOT" \
  --manager systemd

cat <<DONE

QuantTerm canonical host installed at exact SHA $SHA.
Runtime: $RUNTIME_ROOT
Daily broker login when needed: cd "$APP_DIR" && "$PYTHON_BIN" main.py login
Desk: http://<server-ip>:5173
Host status: cd "$APP_DIR" && scripts/quantterm_status.sh --runtime-root "$RUNTIME_ROOT" --manager systemd
DONE
