@echo off
cd /d "%~dp0"
set "APP_PATH=loto_lab\apps\analysis_research_lab.py"
set "PYTHON_EXE="
set "PYTHON_ARGS="

if exist ".\.venv\Scripts\python.exe" (
  ".\.venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
  if not errorlevel 1 set "PYTHON_EXE=.\.venv\Scripts\python.exe"
)

if not defined PYTHON_EXE (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import streamlit" >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
  )
)

if not defined PYTHON_EXE (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -c "import streamlit" >nul 2>nul
    if not errorlevel 1 (
      set "PYTHON_EXE=py"
      set "PYTHON_ARGS=-3"
    )
  )
)

if not defined PYTHON_EXE (
  if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
    "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -c "import streamlit" >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
  )
)

if not defined PYTHON_EXE (
  echo Streamlitを実行できるPythonが見つかりません。requirements.txtを入れるか、PythonをPATHに追加してからもう一度起動してください。
  pause
  exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% -m streamlit run "%APP_PATH%" --server.port 8501
