# Pinned, network-independent evaluator runtime for BigCodeBench-Hard v0.1.1.
# The task data is mounted by the data plane; this image never downloads it.
FROM python:3.10-slim
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org raw.githubusercontent.com pypi.tuna.tsinghua.edu.cn" \
    PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    PIP_DEFAULT_TIMEOUT=2000 \
    PIP_RETRIES=5
RUN apt-get update && apt-get install -y --no-install-recommends git g++ python3-tk zip unzip procps r-base libgdal-dev libfreetype6-dev libpng-dev pkg-config python3-dev python3-matplotlib && rm -rf /var/lib/apt/lists/*
RUN git -c http.sslVerify=false clone https://github.com/bigcode-project/bigcodebench.git /opt/bigcodebench && cd /opt/bigcodebench && git checkout 09dd993f46c3fbf3a799465bb96d524edcb0b199
RUN pip install --no-cache-dir numpy==1.24.3 pyarrow==14.0.1 && cd /opt/bigcodebench && pip install . --no-deps && pip install --no-cache-dir appdirs fire multipledispatch pqdm tempdir termcolor tqdm tree_sitter tree-sitter-python wget && pip install --no-cache-dir -r /opt/bigcodebench/Requirements/requirements-eval.txt
RUN useradd --create-home --uid 10001 evaluator
USER evaluator
WORKDIR /work
