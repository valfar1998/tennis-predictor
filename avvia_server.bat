@echo off
chcp 65001 >nul
cd /d "%~dp0"
title TENNIS_PREDICTOR

if not exist ".venv\Scripts\python.exe" (
    echo [ERRORE] Ambiente virtuale non trovato in .venv
    echo Crea l'ambiente con: python -m venv .venv
    echo Poi installa: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo  TENNIS_PREDICTOR - avvio server UI
echo  ----------------------------------
echo  Apri il browser su: http://localhost:8502
echo  (football-predictor usa la porta 8501)
echo  Chiudi questa finestra per fermare il server.
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8502

if errorlevel 1 (
    echo.
    echo Il server si e' chiuso con un errore.
    pause
)
