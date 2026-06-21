pushd "%~dp0"
where uv >nul 2>nul
if errorlevel 1 winget install --id=astral-sh.uv -e
if not exist ".venv\Scripts\python.exe" uv venv --python 3.10.11
uv pip install --python ".venv\Scripts\python.exe" --upgrade "torch>=2.6" "torchvision>=0.21" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 uv pip install --python ".venv\Scripts\python.exe" --upgrade "torch>=2.6" "torchvision>=0.21" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ".venv\Scripts\python.exe" --upgrade ffmpeg-python opencv-python tqdm numpy --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu124
uv pip list --python ".venv\Scripts\python.exe"
popd
