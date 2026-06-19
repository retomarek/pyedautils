@echo off
REM ============================================================
REM  pyedautils - Doku lokal bauen und Vorschau oeffnen
REM  Build-Output liegt in docs/_build/ (steht in .gitignore).
REM  Voraussetzung: jupyter-book installiert
REM    (pip install "jupyter-book<2" sphinx-autodoc2)
REM ============================================================
setlocal
cd /d "%~dp0"

echo Baue Jupyter Book Doku (fuehrt ALLE Beispiel-Notebooks neu aus) ...
REM --all erzwingt einen vollstaendigen Rebuild. Ohne das baut Sphinx nur
REM Seiten neu, deren .ipynb sich geaendert hat - reine Code-Aenderungen in
REM pyedautils wuerden die Diagramme sonst NICHT neu rendern.
jupyter-book build docs/ --all
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
