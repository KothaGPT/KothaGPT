#!/bin/bash
"""
KothaGPT Deployment Script
=========================

Deploy KothaGPT model and setup API endpoints for production use.

Usage:
    bash deploy/deploy.sh
    bash deploy/deploy.sh --model models/banglagpt_rlhf --port 8000
"""

set -e  # Exit on any error

# Default configuration
MODEL_PATH="models/banglagpt_rlhf"
PORT=8000
HOST="0.0.0.0"
API_NAME="kotha_api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --api-name)
            API_NAME="$2"
            shift 2
            ;;
        --help)
            echo "KothaGPT Deployment Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH     Path to model directory (default: models/banglagpt_rlhf)"
            echo "  --port PORT      Port for API server (default: 8000)"
            echo "  --host HOST      Host for API server (default: 0.0.0.0)"
            echo "  --api-name NAME  Name for API script (default: kotha_api)"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model models/banglagpt_lora --port 9000"
            echo "  $0 --model models/banglagpt_rlhf"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "🚀 Starting KothaGPT Deployment"
log_info "Model Path: $MODEL_PATH"
log_info "Port: $PORT"
log_info "Host: $HOST"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    log_error "Model directory not found: $MODEL_PATH"
    exit 1
fi

# Check if model files exist
MODEL_FILE="$MODEL_PATH/pytorch_model.bin"
if [ ! -f "$MODEL_FILE" ]; then
    log_warning "Standard model file not found, checking for other formats..."
    # Check for other common model file formats
    if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/adapter_model.bin" ]; then
        log_error "No model files found in: $MODEL_PATH"
        exit 1
    fi
fi

log_success "✅ Model files verified"

# Create deployment directory if it doesn't exist
DEPLOY_DIR="deploy"
mkdir -p "$DEPLOY_DIR"

# Create API script
API_SCRIPT="$DEPLOY_DIR/${API_NAME}.py"

log_info "🔧 Creating API script: $API_SCRIPT"

cat > "$API_SCRIPT" << EOF
#!/usr/bin/env python3
"""
KothaGPT API Server
==================

FastAPI server for KothaGPT model inference.
Provides endpoints for text generation and model information.
"""

import os
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="KothaGPT API",
    description="Bangla Language Model API",
    version="1.0.0"
)

# Global model variables
MODEL = None
TOKENIZER = None
MODEL_PATH = "$MODEL_PATH"

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.2

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    model_info: dict

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global MODEL, TOKENIZER

    try:
        logger.info(f"Loading model from: {MODEL_PATH}")

        # Load tokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Load model
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        logger.info("✅ Model loaded successfully")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "KothaGPT API is running!", "model": "$MODEL_PATH"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "tokenizer_loaded": TOKENIZER is not None
    }

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_path": MODEL_PATH,
        "vocab_size": len(TOKENIZER),
        "model_type": str(type(MODEL).__name__),
        "device": str(MODEL.device) if hasattr(MODEL, 'device') else "unknown"
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from prompt."""
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Tokenize input
        inputs = TOKENIZER(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move to model device
        inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = MODEL.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                do_sample=True,
                pad_token_id=TOKENIZER.eos_token_id
            )

        # Decode generated text
        generated_text = TOKENIZER.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )

        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            model_info={
                "model_path": MODEL_PATH,
                "generation_params": {
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "repetition_penalty": request.repetition_penalty
                }
            }
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="$HOST",
        port=$PORT,
        reload=False,
        log_level="info"
    )
EOF

# Make API script executable
chmod +x "$API_SCRIPT"

log_success "✅ API script created: $API_SCRIPT"

# Create systemd service file (optional)
SERVICE_FILE="/etc/systemd/system/${API_NAME}.service"

if [ -d "/etc/systemd" ]; then
    log_info "🔧 Creating systemd service file..."

    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=KothaGPT API Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/$API_SCRIPT
Restart=always
RestartSec=10
Environment=PYTHONPATH=$(pwd)

[Install]
WantedBy=multi-user.target
EOF

    log_success "✅ Systemd service created: $SERVICE_FILE"
    log_info "Enable with: sudo systemctl enable $API_NAME"
    log_info "Start with: sudo systemctl start $API_NAME"
fi

# Create Docker configuration (optional)
DOCKER_DIR="$DEPLOY_DIR/docker"
mkdir -p "$DOCKER_DIR"

log_info "🐳 Creating Docker configuration..."

# Create Dockerfile
cat > "$DOCKER_DIR/Dockerfile" << EOF
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
COPY $MODEL_PATH ./models/kothagpt/

EXPOSE $PORT
CMD ["python", "$DEPLOY_DIR/$API_NAME.py"]
EOF

# Create docker-compose.yml
cat > "$DOCKER_DIR/docker-compose.yml" << EOF
version: '3.8'
services:
  kothagpt-api:
    build: .
    ports:
      - "$PORT:$PORT"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/kothagpt
    restart: unless-stopped
EOF

log_success "✅ Docker configuration created"

# Create nginx configuration (optional)
NGINX_CONF="$DEPLOY_DIR/nginx.conf"

if command -v nginx &> /dev/null; then
    log_info "🌐 Creating nginx reverse proxy configuration..."

    cat > "$NGINX_CONF" << EOF
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

    log_success "✅ Nginx configuration created: $NGINX_CONF"
    log_info "Copy to /etc/nginx/sites-available/ and enable"
fi

# Create deployment documentation
DOC_FILE="$DEPLOY_DIR/README.md"

cat > "$DOC_FILE" << EOF
# KothaGPT Deployment Guide

## Quick Start

### Using Python directly:
\`\`\`bash
python $API_SCRIPT
\`\`\`

### Using Docker:
\`\`\`bash
cd $DOCKER_DIR
docker-compose up --build
\`\`\`

### Using systemd (Linux):
\`\`\`bash
sudo systemctl enable $API_NAME
sudo systemctl start $API_NAME
sudo systemctl status $API_NAME
\`\`\`

## API Endpoints

- \`GET /\` - API status
- \`GET /health\` - Health check
- \`GET /model/info\` - Model information
- \`POST /generate\` - Generate text

## Configuration

- **Model Path**: \`$MODEL_PATH\`
- **Port**: \`$PORT\`
- **Host**: \`$HOST\`

## Logs

- Application logs: Check systemd journal or Docker logs
- Model loading: Check application startup logs
EOF

log_success "✅ Deployment documentation created"

# Final summary
log_success "🎉 KothaGPT Deployment Complete!"
echo ""
echo "📋 Deployment Summary:"
echo "  • Model: $MODEL_PATH"
echo "  • API Port: $PORT"
echo "  • API Script: $API_SCRIPT"
echo "  • Deployment Dir: $DEPLOY_DIR"
echo ""
echo "🚀 Start the API server:"
echo "  python $API_SCRIPT"
echo ""
echo "📖 See $DOC_FILE for detailed deployment instructions"
echo ""
echo "🔗 Once running, API will be available at: http://$HOST:$PORT"
echo "📚 API documentation: http://$HOST:$PORT/docs"