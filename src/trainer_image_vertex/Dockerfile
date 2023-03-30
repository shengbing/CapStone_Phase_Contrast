FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-8:latest
RUN pip install -U cloudml-hypertune
WORKDIR /app
COPY * /app/

ENTRYPOINT ["python", "task.py"]