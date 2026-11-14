@echo off
setlocal
cd /d "%~dp0"
title Fire Safety Detector

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto :nopython

if exist ".venv\Scripts\python.exe" goto :venv_ok
echo [1/3] Tao moi truong ao .venv ...
%PY% -m venv .venv
if errorlevel 1 goto :fail
:venv_ok

call ".venv\Scripts\activate.bat"

if exist ".venv\.deps_ok" goto :deps_ok
echo [2/3] Cai thu vien tu requirements.txt ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo ok> ".venv\.deps_ok"
:deps_ok

echo [3/3] Khoi dong ung dung ...
python main.py %*
if errorlevel 1 goto :fail
exit /b 0

:nopython
echo [LOI] Khong tim thay Python. Cai Python 3.9+ tu python.org
echo       (nho tick "Add python.exe to PATH" va "tcl/tk and IDLE").
pause
exit /b 1

:fail
echo.
echo [LOI] Co loi xay ra, xem thong bao phia tren.
pause
exit /b 1
