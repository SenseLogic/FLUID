pushd "%~dp0"
where uv >nul 2>nul
if errorlevel 1 winget install --id=astral-sh.uv -e
if not exist ".venv\Scripts\python.exe" uv venv --python 3.12.10
uv pip install --python ".venv\Scripts\python.exe" --upgrade --no-cache -f https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/ "torch==2.9.1+rocm7.2.1" "torchvision==0.24.1+rocm7.2.1" "torchaudio==2.9.1+rocm7.2.1"
uv pip install --python ".venv\Scripts\python.exe" --upgrade ffmpeg-python opencv-python tqdm numpy --index-url https://pypi.org/simple
uv pip list --python ".venv\Scripts\python.exe"
popd
