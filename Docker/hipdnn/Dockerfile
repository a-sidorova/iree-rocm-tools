# Dockerfile for work with hipDNN

# Update ROCm version if needed
FROM rocm/dev-ubuntu-24.04:7.1.1-complete
USER root
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Dubai

# Install dependencies
ARG DEPENDENCIES="apt-utils \
                  build-essential \
                  ca-certificates \
                  ccache \
                  cmake \
                  clang-20 \
                  clang-format-20 \
                  clang-tidy-20 \
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
                  lld-20 \
                  llvm-20-dev \
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
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100 && \
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-20 100 && \
    update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-20 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-20 100

# Install Python API
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH=${PATH}:${VIRTUAL_ENV}/bin
RUN python -m pip install --upgrade pip && \
    python -m pip install pre-commit

# Download repository rocm-libraries
WORKDIR /opt
RUN git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git && \
    cd rocm-libraries && \
    git sparse-checkout init --cone && \
    git sparse-checkout set projects/hipdnn && \
    git checkout develop

# Install pre-commit
RUN cd /opt/rocm-libraries && \
    pre-commit install

# Build hipdnn
RUN cd /opt/rocm-libraries/projects/hipdnn && \
    mkdir build && cd build && \
    cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && \
    ninja && ninja install

# Workspace should be mounted in runtime
WORKDIR /workspace

CMD ["/bin/bash"]
