#!/bin/bash

# Install zstd and pciutils (required by the Ollama installer for
# archive extraction and GPU detection)
sudo apt-get update && sudo apt-get install -y zstd pciutils

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
