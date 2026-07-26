pushd "%~dp0"
echo requires python 3.12.10
echo requires AMD ROCm 7.2.1
python.exe --version
python.exe -m pip install --upgrade pip
python.exe -m pip install --upgrade --no-cache-dir -f https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/ "torch==2.9.1+rocm7.2.1" "torchvision==0.24.1+rocm7.2.1" "torchaudio==2.9.1+rocm7.2.1"
python.exe -m pip install --upgrade ffmpeg-python opencv-python tqdm numpy --index-url https://pypi.org/simple
python.exe -m pip list
popd
