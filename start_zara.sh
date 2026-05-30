#!/bin/bash
source /home/kenny/.bashrc
pkill -f "zara_api" 2>/dev/null
pkill -f "catalog_api" 2>/dev/null
sleep 2
cd /home/kenny/barber_ai/production
ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
PYTHONPATH=/home/kenny/barber_ai:/home/kenny/barber_ai/production \
/home/kenny/barber_ai/venv_sdxl/bin/python3 -m uvicorn zara_api:app \
--host 0.0.0.0 --port 8000 &
PYTHONPATH=/home/kenny/barber_ai:/home/kenny/barber_ai/production \
/home/kenny/barber_ai/venv_sdxl/bin/python3 -m uvicorn catalog_api:app \
--host 0.0.0.0 --port 8003 &
echo "Started"
