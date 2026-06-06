#!/bin/bash
# Start FastAPI backend in the background
# It needs to bind to 127.0.0.1 or 0.0.0.0 so Streamlit can reach it
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --loop asyncio &

# Start Streamlit frontend on port 7860 (Hugging Face Default)
streamlit run src/app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0
