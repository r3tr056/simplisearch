# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV DB_HOST=postgres
ENV DB_PORT=5432
ENV DB_NAME=vector_db
ENV DB_USER=postgres
ENV DB_PASSWORD=password
ENV SERVER_PORT=8080
ENV MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ENV MODEL_CACHE_DIR=/app/models

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libpq-dev \
    libeigen3-dev \
    libcpprest-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.17.0.tgz \
    && mkdir -p /usr/local/include/onnxruntime \
    && cp -r onnxruntime-linux-x64-1.17.0/include/* /usr/local/include/onnxruntime/ \
    && cp -r onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/ \
    && rm -rf onnxruntime-linux-x64-1.17.0* \
    && ldconfig

# Create app directory
WORKDIR /app

# Copy source code and CMake files
COPY . /app/

# Build the application
RUN mkdir build \
    && cd build \
    && cmake .. \
    && make

# Create model cache directory
RUN mkdir -p /app/models

# Expose the server port
EXPOSE 8080

# Start the server
CMD ["./build/simpli_search_server"]