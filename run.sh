#!/bin/bash
set -euo pipefail
cd /home/scott/code/telemuze
source .env
exec ./target/debug/telemuze
