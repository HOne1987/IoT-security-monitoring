FROM python:3.9-slim

WORKDIR /app
COPY detector.py /app/

# Install the necessary data science and web libraries
RUN pip install requests pandas scikit-learn prometheus_client

EXPOSE 8001
CMD ["python", "detector.py"]
