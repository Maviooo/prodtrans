#!/bin/bash
# Generuje polskie nagrania testowe głosem "Zosia" z macOS (16 kHz, mono, wav).
set -euo pipefail
cd "$(dirname "$0")/../tests/fixtures"
tmp=$(mktemp -d)
make() {
  say -v Zosia -o "$tmp/$1.aiff" "$2"
  afconvert -f WAVE -d LEI16@16000 -c 1 "$tmp/$1.aiff" "$1.wav"
  echo "$2" > "$1.txt"
}
make pl_1 "Uwaga, stand by światło dwanaście, za chwilę wchodzi chór."
make pl_2 "Mikrofon numer trzy, proszę podgłośnić monitor na scenie."
rm -rf "$tmp"
