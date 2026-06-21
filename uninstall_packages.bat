pushd "%~dp0"
python.exe -m pip freeze > "%TEMP%\pip_freeze.txt"
findstr /R /I /V "^pip== ^setuptools== ^wheel==" "%TEMP%\pip_freeze.txt" > "%TEMP%\pip_uninstall.txt"
python.exe -m pip uninstall -y -r "%TEMP%\pip_uninstall.txt"
del "%TEMP%\pip_freeze.txt"
del "%TEMP%\pip_uninstall.txt"
popd
