pushd "%~dp0"
echo requires python 3.10.11
echo requires CUDA 12.4
python.exe --version
python.exe -m pip install --upgrade pip
python.exe -m pip install --upgrade "torch>=2.6" "torchvision>=0.21" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 python.exe -m pip install --upgrade "torch>=2.6" "torchvision>=0.21" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cpu
python.exe -m pip install --upgrade ffmpeg-python opencv-python tqdm numpy --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu124
python.exe -m pip list
popd
