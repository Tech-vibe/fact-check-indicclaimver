@echo off
setlocal enabledelayedexpansion

rem Change directory to LLM module folder if executed from root
cd /d "%~dp0"

set MODEL_FILE=.\models\Qwen3-8B-IQ4_XS.gguf

if "%~1" neq "" (
    set MODEL_FILE=%~1
) else if not exist "!MODEL_FILE!" (
    for %%F in (.\models\*.gguf) do (
        set MODEL_FILE=%%F
        goto :FOUND_MODEL
    )
)

:FOUND_MODEL
echo =================================================================
echo  Starting LLM Inference Server (llama-server)
echo  Selected Model: !MODEL_FILE!
echo =================================================================

if not exist "!MODEL_FILE!" (
    echo [ERROR] No GGUF model file found in .\models\
    echo Please download a GGUF model into LLM\models\ and try again.
    pause
    exit /b 1
)

set SERVER_EXE=.\llama.cpp\build\bin\Release\llama-server.exe
if not exist "!SERVER_EXE!" (
    if exist ".\llama-server.exe" (
        set SERVER_EXE=.\llama-server.exe
    )
)

echo Running: !SERVER_EXE! -m "!MODEL_FILE!" --port 8080
"!SERVER_EXE!" -m "!MODEL_FILE!" --port 8080 -c 8192 -np 1 -fa on --reasoning off -ngl 99 -ctk q8_0 -ctv q8_0 -ub 1024
pause



