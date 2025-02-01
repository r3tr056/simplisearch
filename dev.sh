#!/bin/bash

function build() {
    echo "Building Docker images..."
    docker-compose build
}

function start() {
    echo "Starting services..."
    docker-compose up -d
    echo "Waiting for services to start..."
    sleep 5
    echo "Services are ready!"
}

function stop() {
    echo "Stopping services..."
    docker-compose down
}

function logs() {
    docker-compose logs -f
}

function test() {
    echo "Testing the API..."
    
    # Test adding a vector
    curl -X POST http://localhost:8080/api/add \
        -H "Content-Type: application/json" \
        -d '{"key":"test1","text":"This is a test document","metadata":{"source":"test"}}'
    
    echo -e "\n\nWaiting 2 seconds before search test...\n"
    sleep 2
    
    # Test searching
    curl -X POST http://localhost:8080/api/search \
        -H "Content-Type: application/json" \
        -d '{"query":"test document","top_k":5,"threshold":0.6}'
}

function cleanup() {
    echo "Cleaning up volumes..."
    docker-compose down -v
}

case "$1" in
    "build")
        build
        ;;
    "start")
        start
        ;;
    "stop")
        stop
        ;;
    "logs")
        logs
        ;;
    "test")
        test
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        echo "Usage: $0 {build|start|stop|logs|test|cleanup}"
        exit 1
esac