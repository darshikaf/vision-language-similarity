#!/bin/bash

SERVICE_URL="http://vision-similarity-service:8000"
URLS_FILE="/urls.txt"
INTERVAL=2  # seconds between requests

echo "Starting load generation for Vision-Language Similarity Service"
echo "Service URL: $SERVICE_URL"
echo "Request interval: ${INTERVAL}s"

# Wait for service to be ready
echo "Waiting for service to be ready..."
while ! curl -s "$SERVICE_URL/evaluator/health" > /dev/null; do
  echo "Service not ready, waiting..."
  sleep 5
done
echo "Service is ready!"

# Start generating load
while true; do
  while IFS= read -r url_path; do
    # Skip empty lines and comments
    [[ -z "$url_path" || "$url_path" =~ ^#.*$ ]] && continue
    
    full_url="$SERVICE_URL$url_path"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Requesting: $full_url"
    
    # Make the request and capture basic stats
    response=$(curl -s -w "%{http_code}|%{response_time}" -o /dev/null "$full_url")
    http_code=$(echo "$response" | cut -d'|' -f1)
    response_time=$(echo "$response" | cut -d'|' -f2)
    
    echo "  → Response: $http_code (${response_time}s)"
    
    sleep "$INTERVAL"
  done < "$URLS_FILE"
done