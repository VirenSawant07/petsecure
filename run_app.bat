@echo off
rem Starts PetSecure. Run "pip install -r requirements.txt" once first.
cd /d "%~dp0"
python -m streamlit run app.py %*
