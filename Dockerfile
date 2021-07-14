FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

# Copy Dependencies
COPY requirements.txt /
COPY detectron2 /detectron2
COPY detectron2_changes /detectron2_changes

# Prepare Environment (1)
RUN mkdir -p /opt/COSMOS/models_final

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Prepare COSMOS
RUN pip3 install cython numpy && \
    pip3 install -r /requirements.txt

# Patch and Install Detectron
RUN cd /detectron2/ && \
    patch -p1 < /detectron2_changes/0001-detectron2-mod.patch && \
    cd / && python3 -m pip install -e detectron2

# Fix PyCocoTools
RUN pip3 uninstall -y pycocotools && \
    pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# Download spaCy
RUN python3 -m spacy download en && \
    python3 -m spacy download en_core_web_sm

# Copy Source
COPY . /opt/COSMOS

# Setup Environment Variables
ENV COSMOS_BASE_DIR /opt/COSMOS
ENV COSMOS_DATA_DIR /mmsys21cheapfakes

# Start the code
ENTRYPOINT []
CMD ["/opt/COSMOS/start.sh"]
