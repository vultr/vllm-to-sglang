#!/usr/bin/env bash
# Recreate repos/ at the upstream tags referenced by this codebase.
#
# repos/ holds read-only clones of vLLM, SGLang, TensorRT-LLM, and AITER
# that the argument-translation, env-translation, metrics, filter, and
# AITER capture/restore/tune modules grep and read for source of truth.
# The directory is gitignored; this script is the canonical way to
# populate it.
#
# To bump any version, edit the corresponding *_VERSION variable below
# and re-run. Idempotent: if a repo already exists at the pinned ref,
# it's skipped; otherwise the script clones (shallow, depth=1) or
# fetches the new ref into the existing repo and detaches HEAD onto it.
#
# A *_VERSION value is either an upstream tag (e.g. "v0.5.12") or a
# 40-char commit SHA. SHAs are required when the base image ships a
# post-tag commit (typical for the ROCm forks: aiter, sglang). Bare-SHA
# fetches rely on github.com's default uploadpack.allowReachableSHA1InWant.
#
# Usage:
#   scripts/sync-repos.sh                # sync all repos
#   scripts/sync-repos.sh sglang         # sync only the named repo

set -euo pipefail

# === Pinned versions (edit these to bump) =====================================
# Tag or 40-char commit SHA. To verify what a base image actually ships:
#   docker run --rm --entrypoint bash <image> -c \
#     'git -C /sgl-workspace/sglang rev-parse HEAD; \
#      git -C /sgl-workspace/aiter  rev-parse HEAD'
#
# Selection rules for each pin (un-dated SGLang tags, vLLM CUDA/ROCm
# commit parity, AITER higher-version-wins) and the full bump procedure
# live in docs/build-and-deploy.md under "Base image and AITER pin
# policy". Read it before editing these values.
AITER_VERSION="cdcfa833bdf554ca75594c90dde4316ea9b50199"   # v0.1.13
SGLANG_VERSION="127b9e3283f7c2a43234b852ff5c9f1796d53624"  # v0.5.12
TRTLLM_VERSION="v1.3.0rc14"
VLLM_VERSION="ad7125a431e176d4161099480a66f0169609a690"  # v0.21.0
# =============================================================================

cd "$(dirname "$0")/.."

sync_repo() {
    local name="$1" url="$2" ref="$3"
    local dir="repos/$name"

    # 40-char hex => bare commit SHA; anything else is treated as a tag.
    local is_sha=false
    [[ "$ref" =~ ^[0-9a-f]{40}$ ]] && is_sha=true

    fetch_ref() {
        if $is_sha; then
            # Bare-SHA fetch; github.com allows it. No local ref is created,
            # but FETCH_HEAD is populated and the SHA becomes reachable.
            git -C "$dir" fetch --depth 1 origin "$ref"
        else
            git -C "$dir" fetch --depth 1 origin "+refs/tags/$ref:refs/tags/$ref"
        fi
    }

    if [[ -d "$dir/.git" ]]; then
        # Make sure we have the ref locally, then compare HEAD to its SHA.
        # ^{commit} dereferences annotated tags to the target commit; for
        # lightweight tags and bare SHAs it's a no-op.
        fetch_ref 2>/dev/null || fetch_ref
        local target current
        target="$(git -C "$dir" rev-parse "$ref^{commit}")"
        current="$(git -C "$dir" rev-parse HEAD)"
        if [[ "$current" == "$target" ]]; then
            echo "[$name] already at $ref ($target)"
            return
        fi
        echo "[$name] updating $current -> $ref ($target)"
        git -C "$dir" checkout --detach "$target"
    else
        echo "[$name] cloning $url at $ref"
        mkdir -p "$dir"
        git -C "$dir" init -q
        git -C "$dir" remote add origin "$url"
        fetch_ref
        git -C "$dir" checkout --detach "$ref"
    fi
}

declare -a TARGETS
if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
else
    TARGETS=(aiter sglang trtllm vllm)
fi

for target in "${TARGETS[@]}"; do
    case "$target" in
        aiter)
            sync_repo aiter \
                https://github.com/ROCm/aiter.git \
                "$AITER_VERSION"
            ;;
        sglang)
            sync_repo sglang \
                https://github.com/sgl-project/sglang.git \
                "$SGLANG_VERSION"
            ;;
        trtllm)
            sync_repo trtllm \
                https://github.com/NVIDIA/TensorRT-LLM.git \
                "$TRTLLM_VERSION"
            ;;
        vllm)
            sync_repo vllm \
                https://github.com/vllm-project/vllm.git \
                "$VLLM_VERSION"
            ;;
        *)
            echo "unknown target: $target" >&2
            echo "valid targets: aiter, sglang, trtllm, vllm" >&2
            exit 2
            ;;
    esac
done
