#!/bin/sh
# Wait up to 2 minutes until the docker container $1 is running.
for i in $(seq 60); do
  # Running state of the container
  if [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]; then
    # Ready
    exit 0
  fi
  # Try again
  sleep 2
done
# Report and fail (systemd retries)
echo "container $1 is not running" >&2
exit 1
