@echo off
REM Iris_win.bat — Windows equivalent of Iris.sh (Linux).
REM
REM Detects a virtual environment under %USERPROFILE%\venvs\, activates it,
REM and starts the Iris server via setup\start_iris_win.py (waitress-based).
REM
REM Usage: double-click this file, or run it from a command prompt.

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "VENVS_LIST=visredPC visred mezt"
set "ACTIVATED=0"

echo [Iris_win] Checking for virtual environments: %VENVS_LIST%
for %%V in (%VENVS_LIST%) do (
    if exist "%USERPROFILE%\venvs\%%V\Scripts\activate.bat" (
        echo [Iris_win] Found virtual environment: %%V
        call "%USERPROFILE%\venvs\%%V\Scripts\activate.bat"
        set "ACTIVATED=1"
        goto :venv_done
    ) else (
        echo [Iris_win] No activate script in %%V, skipping...
    )
)
:venv_done

if "%ACTIVATED%"=="0" (
    echo [Iris_win] WARNING: No virtual environment found. Ensure dependencies are installed globally or in one of the expected venvs.
)

echo [Iris_win] Starting Iris server...
python setup\start_iris_win.py

echo.
echo Proceso finalizado. Presiona una tecla para cerrar esta ventana.
pause >nul
