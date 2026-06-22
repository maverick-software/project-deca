@echo off
setlocal
cd /d "%~dp0\.."
if not exist C:\tmp\decadic_pytest mkdir C:\tmp\decadic_pytest
set TMP=C:\tmp\decadic_pytest
set TEMP=C:\tmp\decadic_pytest
set TMPDIR=C:\tmp\decadic_pytest
.venv\Scripts\python.exe scripts\run_tests_verbose.py %*
exit /b %ERRORLEVEL%
