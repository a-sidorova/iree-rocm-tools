# Docker container with IREE and ROCm 7.0

This directory contains dockerfiles with [IREE](https://github.com/iree-org/iree) (built from sources or from pypi) compiler with target backend [ROCm 7.0](https://rocm.docs.amd.com/en/docs-7.0.0/what-is-rocm.html).

1. Install Docker (if missed) and add user to the docker group.

   ```bash
   sudo apt install docker.io
   sudo usermod -aG docker ${USER}
   su ${USER}
   ```

1. Build the Docker image (build arguments `--build-arg` are optional: the main branch from upstream repositories are cloned by default).

   ```bash
   docker build -t <IDSID>-iree-rocm-dev:latest .
   ```

1. Run Docker container based on the built image with GPU support.

   ```bash
   docker run --name=<IDSID>-iree-rocm-dev -it --privileged \
              --network=host --device=/dev/kfd --device=/dev/dri --group-add video \
              --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host \
              -v <YOUR_WORKSPACE>:/workspace -v <CCACHE_PATH>:/opt/ccache \
              <IDSID>-iree-rocm-dev
   ```

1. Validate the built environment in the docker container.

   ```bash
   > rocm-smi
   =========================================== ROCm System Management Interface ===========================================
   ===================================================== Concise Info =====================================================
   Device  Node  IDs              Temp        Power     Partitions          SCLK   MCLK    Fan  Perf  PwrCap  VRAM%  GPU%  
                 (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                 
   ========================================================================================================================
   ....
   ========================================================================================================================
   ================================================= End of ROCm SMI Log ==================================================
   ```

   ```
   > hello_world_embedded
   4xf32=1 1.1 1.2 1.3
    * 
   4xf32=10 100 1000 10000
    = 
   4xf32=10 110 1200 13000
   ```
