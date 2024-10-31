#!/bin/bash
set -e

# Parse command line arguments
FRESH=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--fresh) FRESH=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Set image based on fresh flag
if [ "$FRESH" = true ]; then
    IMAGE_NAME="mengkluce/neurondb-fresh:latest"
else
    IMAGE_NAME="mengkluce/neurondb-llama-3.1-8b-instruct:latest"
fi

CONTAINER_NAME="neurondb"

# Check if the image exists locally
if ! docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
    echo "Image not found locally. Pulling $IMAGE_NAME..."
    docker pull $IMAGE_NAME
fi

# Check if a container with the same name is already running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container $CONTAINER_NAME already exists. Stopping and removing..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run the container
echo "Starting NeuronDB container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p 5432:5432 \
    $IMAGE_NAME

echo "NeuronDB is running!"
