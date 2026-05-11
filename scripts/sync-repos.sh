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
# and re-run. Idempotent: if a repo already exists at the pinned tag,
# it's skipped; otherwise the script clones (shallow, depth=1) or
# fetches the new tag into the existing repo and detaches HEAD onto it.
#
# Usage:
#   scripts/sync-repos.sh                # sync all repos
#   scripts/sync-repos.sh sglang         # sync only the named repo

set -euo pipefail

# === Pinned versions (edit these to bump) =====================================
AITER_VERSION="v0.1.9.post1"
SGLANG_VERSION="v0.5.11"
TRTLLM_VERSION="v1.3.0rc14"
VLLM_VERSION="v0.20.2"
# =============================================================================

cd "$(dirname "$0")/.."

sync_repo() {
    local name="$1" url="$2" tag="$3"
    local dir="repos/$name"

    if [[ -d "$dir/.git" ]]; then
        # Make sure we have the tag locally, then compare HEAD to its SHA.
        # ^{commit} dereferences annotated tags to the target commit; for
        # lightweight tags it's a no-op.
        git -C "$dir" fetch --depth 1 origin "+refs/tags/$tag:refs/tags/$tag" 2>/dev/null
        local target current
        target="$(git -C "$dir" rev-parse "$tag^{commit}")"
        current="$(git -C "$dir" rev-parse HEAD)"
        if [[ "$current" == "$target" ]]; then
            echo "[$name] already at $tag ($target)"
            return
        fi
        echo "[$name] updating $current -> $tag ($target)"
        git -C "$dir" checkout --detach "$tag"
    else
        echo "[$name] cloning $url at $tag"
        mkdir -p "$dir"
        git -C "$dir" init -q
        git -C "$dir" remote add origin "$url"
        git -C "$dir" fetch --depth 1 origin "+refs/tags/$tag:refs/tags/$tag"
        git -C "$dir" checkout --detach "$tag"
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
