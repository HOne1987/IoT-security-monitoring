# Use Red Hat Universal Base Image 9 (Minimal) - The latest version
FROM registry.access.redhat.com/ubi9/ubi-minimal:latest

LABEL maintainer="MSc_Student"
LABEL description="Hardened Universal IoT Agent (UBI 9)"

# 1. Install Python 3, Pip, and compiler dependencies
# UBI 9 uses 'python3' (which is 3.9+) instead of 'python39'
RUN microdnf install -y python3 python3-pip python3-devel gcc \
    && microdnf clean all \
    && pip3 install prometheus_client psutil

# 2. Security Hardening: Create non-root user
RUN useradd -r -s /bin/false -u 1001 iotuser

# 3. Application Setup
WORKDIR /app
COPY universal_agent.py /app/

# 4. Permissions
RUN chown -R iotuser:iotuser /app

# 5. Switch to non-root
USER 1001

EXPOSE 8000
CMD ["python3", "universal_agent.py"]
