ARG VLLM_VERSION=latest
FROM vllm/vllm-openai:${VLLM_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG VLLM_VERSION=latest
LABEL vllm.version=${VLLM_VERSION}

RUN mkdir -p /app/models

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080

ENV MODEL_NAME=""
ENV MAX_MODEL_LEN=8192
ENV GPU_MEMORY_UTILIZATION=0.95
ENV TENSOR_PARALLEL_SIZE=1
ENV DTYPE=auto
ENV HOST=0.0.0.0
ENV PORT=8080
ENV EXTRA_ARGS=""

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
