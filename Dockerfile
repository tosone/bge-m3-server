FROM ubuntu:22.04 as fetch

ARG USE_CHINA_MIRROR=false
RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
        sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list; \
        sed -i 's/ports.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y git git-lfs python3-pip

RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi && \
    pip install -U "huggingface_hub[cli]" && \
    huggingface-cli download "BAAI/bge-m3" --repo-type model --local-dir /code --local-dir-use-symlinks False && ls -alh /code

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

COPY --from=fetch /code /bge-m3

ARG USE_CHINA_MIRROR=false
RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list; \
    sed -i 's/ports.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list; \
fi

RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3

WORKDIR /app

COPY . .

ENV BGE_M3_MODEL_NAME /bge-m3

RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi && \
    python3 -m pip install -r requirements.txt

CMD ["python", "server.py"]
