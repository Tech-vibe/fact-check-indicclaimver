@echo off
echo Starting llama-server on GPU with Qwen3-8B-Q4_K_M...
echo Model: .\models\Qwen3-8B-Q4_K_M.gguf
echo Ports: 8080 (c_size: 4096, slots: 1, GPU Layers: 99, FlashAttn: ON, Reasoning: OFF)
echo --------------------------------------------------
.\llama.cpp\build\bin\Release\llama-server.exe -m .\models\Qwen3-8B-Q4_K_M.gguf --port 8080 -c 8192 -np 1 -fa on --reasoning off -ngl 99
pause

