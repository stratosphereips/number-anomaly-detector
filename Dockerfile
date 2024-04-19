FROM cgr.dev/chainguard/python:latest-dev as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user -r requirements.txt \
    && rm -rf /home/nonroot/.cache/pip

FROM cgr.dev/chainguard/python:latest

ARG APP=/var/opt/app/
WORKDIR $APP

# Make sure you update Python version in path
#COPY --from=builder /home/nonroot/.local/lib/python3.12/site-packages /home/nonroot/.local/lib/python3.12/site-packages
COPY --from=builder /home/nonroot/.local /home/nonroot/.local

COPY number_anomaly_detector.py $APP/
COPY test-numbers.txt $APP/

CMD ["python", "/var/opt/app/number_anomaly_detector.py"]

