FROM nvidia/cuda:12.6.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config \
    cmake ninja-build git curl wget python3 python3-pip \
    ca-certificates gnupg lsb-release libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/miniforge3
RUN wget --quiet "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /tmp/miniforge3.sh \
    && /bin/bash /tmp/miniforge3.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge3.sh \
    && $CONDA_DIR/bin/conda clean -afy

ENV PATH="${CONDA_DIR}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN $CONDA_DIR/bin/conda install -y -n base -c conda-forge mamba
RUN conda init

WORKDIR /workspace/attention-influence
CMD ["/bin/bash", "-lc"]

