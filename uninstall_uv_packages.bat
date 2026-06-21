pushd "%~dp0"
rmdir /s /q ".venv" 2>nul
rmdir /s /q "%LOCALAPPDATA%\uv\cache" 2>nul
rmdir /s /q "%USERPROFILE%\.cache\uv" 2>nul
popd
