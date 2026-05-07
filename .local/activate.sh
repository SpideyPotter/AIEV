# Source this in future pod sessions:  source /workspace/.local/activate.sh
PREFIX=/workspace/.local
PY=python3.10
export PYTHONPATH="$PREFIX/lib/$PY/site-packages:$PREFIX/local/lib/$PY/dist-packages:$PREFIX/lib/$PY/dist-packages:${PYTHONPATH:-}"
export PATH="$PREFIX/bin:$PREFIX/local/bin:$PATH"
