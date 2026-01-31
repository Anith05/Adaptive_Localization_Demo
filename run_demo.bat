@echo off
REM ===================================================================
REM Adaptive Robot Localization Demo Launcher (Windows)
REM ===================================================================

echo.
echo ============================================================
echo    ADAPTIVE ROBOT LOCALIZATION DEMO - LAUNCHER
echo ============================================================
echo.

cd /d "%~dp0"

echo Checking setup...
python verify_setup.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Setup verification failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    STARTING SIMULATION
echo ============================================================
echo.
echo Controls:
echo   - SPACEBAR: Pause/Resume
echo   - ESC or Close Window: Exit
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak > nul

python sim\run_simulation.py

echo.
echo ============================================================
echo    SIMULATION ENDED
echo ============================================================
echo.
pause
