FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED=1

# Use apt in docker best practices, see https://docs.docker.com/reference/dockerfile/#example-cache-apt-packages.
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    # git is needed for dependencies specified as git repositories
    git \
    # ca-certificates is needed to verify the authenticity of servers hosting dependencies
    ca-certificates \
    # curl is needed for our preferred uv installation method
    curl \
    # mesa-utils is needed for minetest
    mesa-utils

# RUN apt-get update && apt-get install -y \
#   ffmpeg git vim curl software-properties-common \
#   libglew-dev x11-xserver-utils xvfb \
#   && apt-get clean

RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=v0.33.0 bash
ENV PATH="$PATH:/root/.pixi/bin"

# Workdir
WORKDIR /workspace

# Copy project meta data only.
COPY --link pixi.lock pyproject.toml ./

# Required for editable install to work.
RUN mkdir dreamerv3 embodied

# Should use a secret instead https://docs.docker.com/build/building/secrets/.
RUN --mount=type=cache,target=/root/.cache/rattler,sharing=locked \
    pixi auth login https://repo.prefix.dev --token pfx-AwGAN1WWoDR82UwU51ei2DMSvIgrwrlroUq9 && \
    pixi install && \
    mkdir -p .pixi/shell-hooks && \
    pixi shell-hook --shell bash > .pixi/shell-hooks/default

ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

ENTRYPOINT ["/bin/bash", "-l", "/workspace/entrypoint.sh"]

CMD ["bash"]

# Source
COPY --link . .

# Cloud
ENV GCS_RESOLVE_REFRESH_SECS=60
ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600
RUN chown 1000:root . && chmod 775 .
