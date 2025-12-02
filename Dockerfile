FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FastAPI runs on port 8000, Streamlit on 8501
EXPOSE 8000 8501

CMD ["bash", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501"]
