FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Install necessary packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-dev \
        software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install torch torchvision

# Install other Python packages
RUN pip3 install chess pytorch-lightning

# Download the Lichess games archive
RUN curl -L -O https://database.lichess.org/standard/lichess_db_standard_rated_2022-02.pgn.bz2 && \
    bunzip2 lichess_db_standard_rated_2022-02.pgn.bz2

# Define a PyTorch dataset for the chess games
COPY chess_dataset.py /

# Define a PyTorch model for chess
COPY chess_model.py /

# Train the chess model with PyTorch Lightning
CMD ["python3", "-m", "lightning_fit"]
