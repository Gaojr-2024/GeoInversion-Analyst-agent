@echo off
setlocal

echo Setting up Git environment...
set "GIT_PATH=C:\Program Files\Git\cmd\git.exe"

if not exist "%GIT_PATH%" (
    echo Git not found at default location. Trying to find it...
    for /f "tokens=*" %%i in ('where git') do set "GIT_PATH=%%i"
)

if not exist "%GIT_PATH%" (
    echo Error: Git not found! Please install Git.
    pause
    exit /b 1
)

echo Using Git at: "%GIT_PATH%"

echo Configuring network settings...
"%GIT_PATH%" config --global http.postBuffer 524288000
"%GIT_PATH%" config --global http.sslVerify false

echo Pushing to GitHub...
:retry
"%GIT_PATH%" push -u origin main
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Push failed. Retrying in 5 seconds...
    timeout /t 5
    goto retry
)

echo.
echo Upload successful!
pause
