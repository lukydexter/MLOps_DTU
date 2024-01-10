# Base image
FROM --platform=linux/amd64 python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/


WORKDIR /
RUN pip install . --no-cache-dir 

ENTRYPOINT ["python", "-u", "src/train_model.py"]