@echo off
echo ===================================================
echo  Starting IndicClaimVerifier GPU Pipeline
echo ===================================================

:: 1. Kill any existing llama-server to prevent port 8080 conflicts
taskkill /IM llama-server.exe /F >nul 2>&1

:: 2. Start llama-server in the background (hidden)
echo Starting llama-server on GPU in background...
start /B "" ".\llama.cpp\build\bin\Release\llama-server.exe" -m ".\models\Qwen3-8B-Q4_K_M.gguf" --port 8080 -c 8192 -np 1 -fa on --reasoning off -ngl 99 > nul 2>&1

:: 3. Wait 12 seconds for the model to fully load into GPU VRAM
echo Waiting for GPU server to initialize...
timeout /t 12 /nobreak > nul

:: 4. Run the python pipeline using the virtual environment
echo Running verification pipeline...
call .venv\Scripts\python.exe llm_pipeline.py

:: 5. Clean up the server process once python finishes
echo Shutdown background server...
taskkill /IM llama-server.exe /F >nul 2>&1

echo ===================================================
echo  Pipeline execution complete!
echo ===================================================
pause
