# Dockerfile for development of LLVM

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

# Clone LLVM-project
ARG LLVM_UPSTREAM_REPO_URL="https://github.com/llvm/llvm-project.git"
ARG LLVM_REPO_URL="https://github.com/a-sidorova/llvm-project.git"
ARG LLVM_BRANCH="main"
ARG LLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld;mlir;polly"
ARG LLVM_TARGETS_TO_BUILD="X86;AMDGPU"

ENV LLVM_SRC=/opt/llvm-src \
    LLVM_BUILD=/opt/llvm-build \
    LLVM_INSTALL=/opt/llvm
RUN mkdir -p ${LLVM_SRC} ${LLVM_BUILD} ${LLVM_INSTALL} && \
    git lfs install --skip-repo
RUN git clone -b ${LLVM_BRANCH} ${LLVM_REPO_URL} ${LLVM_SRC} && \
    cd ${LLVM_SRC} && git submodule update --init --recursive
RUN git remote add upstream ${LLVM_UPSTREAM_REPO_URL} && git fetch upstream

# Configure and build LLVM (https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
RUN cmake -S ${LLVM_SRC}/llvm -B ${LLVM_BUILD} \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL} \
          -DCMAKE_C_COMPILER=clang \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DLLVM_USE_LINKER=lld \
          -DLLVM_ENABLE_CCACHE=ON \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DLLVM_ENABLE_PROJECTS=${LLVM_ENABLE_PROJECTS} \
          -DLLVM_TARGETS_TO_BUILD=${LLVM_TARGETS_TO_BUILD}
RUN cmake --build ${LLVM_BUILD} -j $(nproc) && \
    cmake --build ${LLVM_BUILD} --target install -j $(nproc)

# Update environment variables
ENV PATH=${PATH}:${LLVM_INSTALL}/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LLVM_INSTALL}/lib

# Workspace should be mounted in runtime
WORKDIR /workspace

CMD ["/bin/bash"]
