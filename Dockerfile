FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODELS_DIR=/models \
    API_PORT=8099 \
    COMFYUI_REF=ce1e922d8623bb507180e32a0c01a7d0e29edef7

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

WORKDIR /opt/ComfyUI
RUN git init \
    && git remote add origin https://github.com/Comfy-Org/ComfyUI.git \
    && git fetch --depth 1 origin ${COMFYUI_REF} \
    && git checkout --detach FETCH_HEAD \
    && pip install -r requirements.txt \
    && pip install huggingface_hub

COPY app /opt/flux3/app
COPY workflows /opt/flux3/workflows
COPY custom_nodes/optional_image_batch.py /opt/ComfyUI/custom_nodes/optional_image_batch.py
COPY entrypoint.sh /opt/flux3/entrypoint.sh
RUN chmod +x /opt/flux3/entrypoint.sh \
    && mkdir -p /models/diffusion_models /models/text_encoders /models/vae /output /input \
    && rm -rf models output input \
    && ln -s /models /opt/ComfyUI/models \
    && ln -s /output /opt/ComfyUI/output \
    && ln -s /input /opt/ComfyUI/input

EXPOSE 8099
VOLUME ["/models"]
ENTRYPOINT ["/opt/flux3/entrypoint.sh"]
