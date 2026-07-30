@echo off
echo Starting llama-server on GPU with Qwen3-8B-IQ4_XS...
echo Model: .\models\Qwen3-8B-IQ4_XS.gguf
echo Ports: 8080 (c_size: 8192, slots: 1, GPU Layers: 99, FlashAttn: ON, Reasoning: OFF, KV: Q8_0, MicroBatch: 1024)
echo --------------------------------------------------
.\llama.cpp\build\bin\Release\llama-server.exe -m .\models\Qwen3-8B-IQ4_XS.gguf --port 8080 -c 8192 -np 1 -fa on --reasoning off -ngl 99 -ctk q8_0 -ctv q8_0 -ub 1024
pause


