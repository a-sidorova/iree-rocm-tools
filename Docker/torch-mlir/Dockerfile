# Dockerfile for development of torch-mlir

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

# Use cland and ccache
ENV CCACHE_DIR=/opt/ccache \
    CCACHE_MAXSIZE=30G \
    CCACHE_COMPRESS=1
ENV CC=/usr/lib/ccache/clang \
    CXX=/usr/lib/ccache/clang++ \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Clone torch-mlir project
ARG TM_UPSTREAM_REPO_URL="https://github.com/llvm/torch-mlir.git"
ARG TM_REPO_URL="https://github.com/a-sidorova/torch-mlir.git"
ARG TM_BRANCH="main"

ENV TM_SRC=/opt/torch-mlir-src \
    TM_BUILD=/opt/torch-mlir-build
RUN mkdir -p ${TM_SRC} ${TM_BUILD} && \
    git lfs install --skip-repo
RUN git clone -b ${TM_BRANCH} ${TM_REPO_URL} ${TM_SRC} && \
    cd ${TM_SRC} && git submodule update --init --recursive
RUN cd ${TM_SRC} && git remote add upstream ${TM_UPSTREAM_REPO_URL}

# Install Python API
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH=${PATH}:${VIRTUAL_ENV}/bin
RUN python -m pip install --upgrade pip
RUN python -m pip install -r ${TM_SRC}/requirements.txt && \
    python -m pip install -r ${TM_SRC}/build-requirements.txt
RUN python -m pip install -r ${TM_SRC}/pytorch-requirements.txt && \
    python -m pip install -r ${TM_SRC}/torchvision-requirements.txt
RUN python -m pip install -r ${TM_SRC}/test-requirements.txt

# Configure and build the project "in-tree"
RUN cmake -S ${TM_SRC}/externals/llvm-project/llvm -B ${TM_BUILD} \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DCMAKE_C_COMPILER=clang \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DPython3_FIND_VIRTUALENV=ONLY \
          -DPython_FIND_VIRTUALENV=ONLY \
          -DPython3_EXECUTABLE=${VIRTUAL_ENV}/bin/python \
          -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
          -DLLVM_TARGETS_TO_BUILD=host \
          -DLLVM_ENABLE_PROJECTS=mlir \
          -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
          -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=${TM_SRC} \
          -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
          -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON
RUN cmake --build ${TM_BUILD} -j $(nproc)

# Update environment variables
ENV PATH=${PATH}:${TM_BUILD}/bin

# Workspace should be mounted in runtime
WORKDIR /workspace

CMD ["/bin/bash"]
