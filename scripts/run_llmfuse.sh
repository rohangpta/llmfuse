#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)

IMAGE_NAME=${IMAGE_NAME:-llmfuse-fs}
MOUNT_DIR=${MOUNT_DIR:-${REPO_ROOT}/mount}

echo "🔧 Building ${IMAGE_NAME} from ${REPO_ROOT}/Dockerfile.llmfuse..."
docker build -t "${IMAGE_NAME}" -f "${REPO_ROOT}/Dockerfile.llmfuse" "${REPO_ROOT}"

mkdir -p "${MOUNT_DIR}"

echo "📂 Mount directory: ${MOUNT_DIR}"
echo "🌐 Remote endpoint: ${LLMFUSE_REMOTE_ENDPOINT:-<not set>}"

docker run \
  --rm \
  --privileged \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  --security-opt apparmor:unconfined \
  -e LLMFUSE_REMOTE_ENDPOINT="${LLMFUSE_REMOTE_ENDPOINT:-}" \
  -e LLMFUSE_REMOTE_TOKEN="${LLMFUSE_REMOTE_TOKEN:-}" \
  -v "${MOUNT_DIR}:/mnt/llmfuse" \
  "${IMAGE_NAME}"
