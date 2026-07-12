@echo off
echo ========================================================
echo               EdgePilot Orchestrator Launcher
echo ========================================================
echo.

:: Set the PATH globally for this batch execution context so all child processes inherit it
set "PATH=C:\msys64\mingw64\bin;%PATH%"

echo 1. Starting C++ Daemon (Port 12345)...
start "EdgePilot C++ Daemon" cmd /k "build_test\edgepilot_daemon.exe"

echo 2. Starting FastAPI Gateway (Port 8000)...
start "EdgePilot FastAPI Gateway" cmd /k "python -m uvicorn api.main:app --port 8000"

echo 3. Waiting for servers to initialize...
timeout /t 3 /nobreak > nul

echo 4. Opening Dashboard in default browser...
start http://localhost:8000/dashboard/

echo.
echo EdgePilot processes successfully spawned in new windows!
echo To shut down, simply close the respective command prompt windows.
echo.
pause
