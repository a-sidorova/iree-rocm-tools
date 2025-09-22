# Dockerfile for work with IREE + ROCm

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

# Clone IREE
ARG IREE_REPO_URL="https://github.com/iree-org/iree"
ARG IREE_BRANCH="main"

ENV IREE_SRC=/opt/iree-src \
    IREE_BUILD=/opt/iree-build \
    IREE_INSTALL=/opt/iree-install
RUN mkdir -p ${IREE_SRC} ${IREE_BUILD} ${IREE_INSTALL} && \
    git lfs install --skip-repo
RUN git clone -b ${IREE_BRANCH} ${IREE_REPO_URL} ${IREE_SRC} && \
    cd ${IREE_SRC} && git submodule update --init --recursive

# Install Python requirements
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH=${PATH}:${VIRTUAL_ENV}/bin
RUN python -m pip install --upgrade pip && \
    python -m pip install -r ${IREE_SRC}/runtime/bindings/python/iree/runtime/build_requirements.txt

# Configure and build IREE (https://iree.dev/building-from-source/getting-started/)
RUN cmake -S ${IREE_SRC} -B ${IREE_BUILD} \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DCMAKE_INSTALL_PREFIX=${IREE_INSTALL} \
          -DCMAKE_C_COMPILER=clang \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DLLVM_USE_LINKER=lld \
          -DIREE_ENABLE_CCACHE=ON \
          -DIREE_BUILD_COMPILER=ON \
          -DIREE_BUILD_PYTHON_BINDINGS=ON \
          -DPython3_EXECUTABLE=$(which python) \
          -DIREE_BUILD_BINDINGS_TFLITE=OFF, \
          -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF, \
          -DIREE_HAL_DRIVER_HIP=ON, \
          -DIREE_TARGET_BACKEND_ROCM=ON
RUN cmake --build ${IREE_BUILD} -j $(nproc) && \
    cmake --build ${IREE_BUILD} --target install -j $(nproc)

# Update environment variables
ENV PATH=${PATH}:${IREE_INSTALL}/bin
ENV PYTHONPATH=${PYTHONPATH}:${IREE_BUILD}/compiler/bindings/python/:${IREE_BUILD}/runtime/bindings/python

# Validate
RUN python -c "import iree.compiler; help(iree.compiler)"
RUN python -c "import iree.runtime; help(iree.runtime)"

# Clone and build IREE-turbine Python API
ARG IREE_TURBINE_REPO_URL="https://github.com/iree-org/iree-turbine.git"
ARG IREE_TURBINE_BRANCH="main"

ENV IREE_TURBINE_SRC=/opt/iree-turbine-src
RUN git clone -b ${IREE_TURBINE_BRANCH} ${IREE_TURBINE_REPO_URL} ${IREE_TURBINE_SRC} && \
    cd ${IREE_TURBINE_SRC} && git submodule update --init --recursive
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN cd ${IREE_TURBINE_SRC} && python ./setup.py install

# Validate
RUN python -c "import iree.turbine; help(iree.turbine)"

# Workspace should be mounted in runtime
WORKDIR /workspace

CMD ["/bin/bash"]
