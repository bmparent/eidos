@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 local_server.py --open-browser
) else (
  python local_server.py --open-browser
)
echo.
echo Sentinel stopped. If Python was not found, install Python 3.11 or newer.
pause
