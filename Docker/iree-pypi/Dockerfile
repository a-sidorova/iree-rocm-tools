# Dockerfile for work with Pithon IREE + ROCm + SharkAI

# Update ROCm version if needed
FROM rocm/dev-ubuntu-24.04:7.0-complete
USER root
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Dubai

# Install dependencies
ARG DEPENDENCIES="apt-utils \
                  build-essential \
                  ca-certificates \
                  ccache \
                  cmake \
                  clang-18 \
                  curl \
                  git \
                  git-lfs \
                  libzstd-dev \
                  libxml2-dev \
                  libedit-dev \
                  libtinfo-dev \
                  libncurses5-dev \
                  libssl-dev \
                  libcurl4-openssl-dev \
                  lld-18 \
                  llvm-18-dev \
                  nano \
                  ninja-build \
                  pkg-config \
                  python3 \
                  python3-dev \
                  python3-pip \
                  python3-setuptools \
                  python3-venv \
                  python3-wheel \
                  wget \
                  unzip \
                  zlib1g-dev"
RUN apt-get update && \
    apt-get install -y -qq --no-install-recommends ${DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/*

# Make clang/lld default
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-18 100

# Install Python API
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH=${PATH}:${VIRTUAL_ENV}/bin
RUN python -m pip install --upgrade pip
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install iree-base-compiler iree-base-runtime iree-turbine shark-ai[apps]

# Validate
RUN python -c "import iree.compiler; help(iree.compiler)"
RUN python -c "import iree.runtime; help(iree.runtime)"
RUN python -c "import iree.turbine; help(iree.turbine)"
RUN python -m pip install transformers diffusers wave_lang

# Workspace should be mounted in runtime
WORKDIR /workspace

CMD ["/bin/bash"]
