FROM python:3.11-slim
WORKDIR /app
RUN pip install streamlit==1.40.2 requests==2.32.3
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
