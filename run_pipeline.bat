@echo off
setlocal

:: Check if --repair argument was passed
set "MODE=NORMAL"
set "REPAIR_ID="
for %%A in (%*) do (
    if /I "%%A"=="--repair" set "MODE=REPAIR"
)

echo ===================================================
if "%MODE%"=="REPAIR" (
    echo  IndicClaimVerifier - REPAIR MODE
) else (
    echo  Starting IndicClaimVerifier GPU Pipeline
)
echo ===================================================

:: 1. Kill any existing llama-server to prevent port 8080 conflicts
taskkill /IM llama-server.exe /F >nul 2>&1

:: 2. Start llama-server in the background (hidden)
echo Starting llama-server on GPU in background...
start /B "" ".\llama.cpp\build\bin\Release\llama-server.exe" -m ".\models\Qwen3-8B-IQ4_XS.gguf" --port 8080 -c 8192 -np 1 -fa on --reasoning off -ngl 99 -ctk q8_0 -ctv q8_0 -ub 1024 > nul 2>&1

:: 3. Wait 12 seconds for the model to fully load into GPU VRAM
echo Waiting for GPU server to initialize...
timeout /t 12 /nobreak > nul

:: 4. Run the python pipeline, passing through all arguments (%*)
::    Normal mode : run_pipeline.bat
::    Repair mode : run_pipeline.bat --repair "S2/T/BN/1045"
echo Running pipeline with args: %*
call .venv\Scripts\python.exe llm_pipeline.py %*

:: 5. Clean up the server process once python finishes
echo Shutting down background server...
taskkill /IM llama-server.exe /F >nul 2>&1

echo ===================================================
if "%MODE%"=="REPAIR" (
    echo  Repair complete!
) else (
    echo  Pipeline execution complete!
)
echo ===================================================
pause
endlocal
