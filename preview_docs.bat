@echo off
REM ============================================================
REM  pyedautils - Doku lokal bauen und Vorschau oeffnen
REM  Build-Output liegt in docs/_build/ (steht in .gitignore).
REM  Voraussetzung: jupyter-book installiert
REM    (pip install "jupyter-book<2" sphinx-autodoc2)
REM ============================================================
setlocal
cd /d "%~dp0"

echo Baue Jupyter Book Doku (fuehrt die Beispiel-Notebooks aus) ...
jupyter-book build docs/
if errorlevel 1 (
    echo.
    echo [FEHLER] Build fehlgeschlagen - siehe Meldungen oben.
    pause
    exit /b 1
)

echo.
echo Oeffne Vorschau im Browser ...
start "" "docs\_build\html\index.html"
endlocal
