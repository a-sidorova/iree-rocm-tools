# Dockerfile for work with Fusili

# Use built iree image (iree-with-rocm directory)
FROM asidorov-iree-rocm-dev

# Install dependencies
ARG DEPENDENCIES="catch2 \
                  clang-format"
RUN apt-get update && \
    apt-get install -y -qq --no-install-recommends ${DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/*

# Clone repository
ARG FUSILLI_REPO_URL="https://github.com/a-sidorova/fusilli.git"
ARG FUSILLI_REPO_UPSTREAM_URL="https://github.com/iree-org/fusilli.git"
ARG FUSILLI_BRANCH="main"

ENV FUSILLI_SRC=/opt/fusilli-src \
    FUSILLI_BUILD=/opt/fusilli-build
RUN mkdir -p ${FUSILLI_SRC} ${FUSILLI_BUILD} && \
    git lfs install --skip-repo
RUN git clone -b ${FUSILLI_BRANCH} ${FUSILLI_REPO_URL} ${FUSILLI_SRC} && \
    cd ${FUSILLI_SRC} && \
    git remote add upstream ${FUSILLI_REPO_UPSTREAM_URL} && \
    git submodule update --init --recursive

RUN python -m pip install --upgrade pip && \
    python -m pip install lit

# Configure Fusilli
# `mkdir -p /dev/kfd` - Hand-hock to support AMGPU Backend
RUN mkdir -p /dev/kfd && \
          cmake -S ${FUSILLI_SRC} -B ${FUSILLI_BUILD} \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DCMAKE_C_COMPILER=clang \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DIREERuntime_DIR=${IREE_BUILD}/lib/cmake/IREE
RUN cmake --build ${FUSILLI_BUILD} -j $(nproc) && \
    cmake --build ${FUSILLI_BUILD} --target all -j $(nproc)

# Workspace should be mounted in runtime
WORKDIR /workspace

CMD ["/bin/bash"]
