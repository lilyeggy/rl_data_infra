FROM node:22-bookworm-slim

ARG PI_VERSION=0.84.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates git \
    && npm install --global "@earendil-works/pi-coding-agent@${PI_VERSION}" --no-fund --no-audit \
    && rm -rf /var/lib/apt/lists/* /root/.npm

COPY pi_data_plane_wrapper.py /opt/agent-data-plane/pi_data_plane_wrapper.py

WORKDIR /workspace
CMD ["python3", "/opt/agent-data-plane/pi_data_plane_wrapper.py", "--help"]
