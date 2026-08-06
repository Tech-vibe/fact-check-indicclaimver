@echo off
setlocal

:: Check if --repair argument was passed
set "MODE=NORMAL"
for %%A in (%*) do (
    if /I "%%A"=="--repair" set "MODE=REPAIR"
)

echo ===================================================
if "%MODE%"=="REPAIR" (
    echo  IndicClaimVerifier - REPAIR MODE
) else (
    echo  Starting IndicClaimVerifier LLM Module
)
echo ===================================================

:: Run python pipeline, passing through all arguments (%*)
echo Executing LLM Pipeline with args: %*
call .venv\Scripts\python.exe llm_pipeline.py %*

echo ===================================================
if "%MODE%"=="REPAIR" (
    echo  Repair complete!
) else (
    echo  Pipeline execution complete!
)
echo ===================================================
pause
endlocal

