@echo off
cd /d "%~dp0"
".\.venv\Scripts\streamlit.exe" run analysis_research_lab.py --server.port 8501
