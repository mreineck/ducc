FROM debian:testing-slim
RUN apt-get update && apt-get install -y git python3-pip python3-pytest clang && rm -rf /var/lib/apt/lists/*
