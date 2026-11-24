#!/bin/zsh
set -eu
cd "${0:A:h}"
url='http://127.0.0.1:8797'
if curl -fsS --max-time 2 "$url/api/session" | /usr/bin/grep -q '"token"'; then
  open "$url"
  exit 0
fi
p=/opt/homebrew/bin/python3
[[ -x "$p" ]] || p="$(command -v python3)"
"$p" -u server.py &
labpid=$!
trap 'kill "$labpid" 2>/dev/null || true' EXIT
for attempt in {1..30}; do
  if curl -fsS --max-time 1 "$url/api/session" >/dev/null 2>&1; then open "$url"; break; fi
  sleep .2
done
echo '实验台在本机运行。请保留此窗口；录制时也请保留发起录制的浏览器页。'
wait "$labpid"
