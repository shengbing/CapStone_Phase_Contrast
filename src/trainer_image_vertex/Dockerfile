FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U cloudml-hypertune  pandas==0.24.2
WORKDIR /app
COPY model.py .
COPY task.py .

ENTRYPOINT ["python", "task.py"]