FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest
RUN pip install -U cloudml-hypertune  pandas==0.24.2 tensorflow==2.8.0
WORKDIR /app
COPY * .

ENTRYPOINT ["python", "task.py"]