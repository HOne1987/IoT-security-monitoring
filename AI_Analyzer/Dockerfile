FROM python:3.11-slim

WORKDIR /app/code

# Install the necessary data science and web libraries
RUN pip install requests pandas scikit-learn==1.8.0 prometheus_client

EXPOSE 8001
CMD ["python", "detector.py"]
