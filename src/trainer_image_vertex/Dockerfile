FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest
RUN pip install -U cloudml-hypertune
WORKDIR /app
COPY * .

ENTRYPOINT ["python", "task.py"]