FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U cloudml-hypertune  pandas==0.24.2 tensorflow==2.8.0
WORKDIR /app
COPY * .

ENTRYPOINT ["python", "task.py"]