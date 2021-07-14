FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

ENV COSMOS_BASE_DIR /opt/COSMOS_ws
ENV COSMOS_DATA_DIR /mmsys21cheapfakes

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Prepare Environment (1)
RUN mkdir -p /opt/COSMOS && mkdir -p /opt/COSMOS_ws/models_final

# Copy Source
COPY . /opt/COSMOS

# Prepare Environment (2)
RUN mkdir -p /opt/COSMOS/models

# TODO: Copy checkpoint to appropriate location

# Prepare COSMOS
RUN cd /opt/COSMOS && \
    pip3 install cython numpy && \
    pip3 install -r requirements.txt

# Download spaCy
RUN python3 -m spacy download en && \
    python3 -m spacy download en_core_web_sm

# Patch and Install Detectron
RUN cd /opt/COSMOS/detectron2/ && \
    patch -p1 < ../detectron2_changes/0001-detectron2-mod.patch && \
    cd .. && python3 -m pip install -e detectron2

# Fix PyCocoTools
RUN pip3 uninstall -y pycocotools && \
    pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# Start the code
COPY start.sh /
ENTRYPOINT []
CMD ["/start.sh"]
