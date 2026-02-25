@echo off
setlocal enabledelayedexpansion

REM Execution timeout (seconds)
set TIMEOUT=300

REM Collect all tutorial source files
set PY_TUTORIALS=
for %%f in (tutorials\src\tutorial*.py) do (
    set PY_TUTORIALS=!PY_TUTORIALS! "%%f"
)
REM Compute corresponding .ipynb paths
set IPYNBS=
for %%f in (%PY_TUTORIALS%) do (
    set base=%%~nf
    REM Remove tutorial prefix if needed; keep same basename
    set IPYNBS=!IPYNBS! "tutorials\!base!.ipynb"
)
goto :start

:process_one
REM %1 = input .py file
set INFILE=%1
set BASENAME=%~n1
set OUTFILE=tutorials\%BASENAME%.ipynb
echo Converting "%INFILE%" -> "%OUTFILE%"
jupytext --to notebook "%INFILE%" --output "%OUTFILE%"
echo Executing "%OUTFILE%"
jupyter nbconvert ^
  --execute ^
  --to notebook ^
  --inplace ^
  --ExecutePreprocessor.timeout=%TIMEOUT% ^
  "%OUTFILE%"
goto :eof


:tutorials
echo Running tutorial pipeline...
for %%f in (%PY_TUTORIALS%) do (
    call :process_one "%%~f"
)
echo.
echo All tutorials converted and executed successfully.
goto :eof

:html
call :tutorials
echo Rendering HTML previews
jupyter nbconvert --to html %IPYNBS%
goto :eof

:pdf
call :tutorials
echo Creating PDF directory
if not exist tutorials\pdf mkdir tutorials\pdf

echo Converting notebooks to PDF...

for %%f in (%IPYNBS%) do (
    echo Converting %%f → PDF
    jupyter nbconvert --to pdf "%%f" --output-dir tutorials\pdf
)

echo.
echo All PDFs generated successfully.
goto :eof

:clean
echo Cleaning HTML files...
del /q tutorials\*.html 2>nul
goto :eof

:distclean
echo Removing generated notebooks and HTML...
del /q tutorials\*.ipynb 2>nul
del /q tutorials\*.html 2>nul
goto :eof

:start
if "%~1"=="" (
    echo Usage
    echo   make.bat tutorials
    echo   make.bat html
    echo   make.bat pdf
    echo   make.bat clean
    echo   make.bat distclean
    goto :eof
)
call :%1
:eof