#!/usr/bin/env bash
# Apply local vllm-shim patches to an upstream repo checkout.
#
# Sets up the branch model the rebuild script expects:
#
#   refs/vllm-shim/upstream            <- the upstream-pinned commit
#       |
#       +-- patched/base               <- base/*.patch applied as commits
#             |
#             +-- patched/<platform>   <- <platform>/*.patch on top
#
# For mono-platform repos (no base/ subdir under patches/<repo>/), the
# script skips patched/base and roots patched/<platform> directly on
# refs/vllm-shim/upstream.
#
# This script runs in two places: inside the Docker build (against a
# clean clone the base image provides) and on the dev's repos/<repo>/
# checkout. In both cases it is idempotent when nothing changed; if the
# upstream-pinned ref no longer matches HEAD or patched branches already
# exist, it rebuilds them.
#
# Author identity for the replayed commits is fixed via env so rebuild
# output (via git format-patch) stays byte-stable across runs.
#
# Usage:
#   apply-patches.sh <repo> <platform> <target-dir>
#
# Args:
#   repo         logical repo name; selects the patches/<repo>/ tree
#   platform     "rocm" | "cuda" | ...; matches patches/<repo>/<platform>/
#   target-dir   filesystem path to the upstream checkout to patch
#
# Examples:
#   apply-patches.sh aiter  rocm /sgl-workspace/aiter
#   apply-patches.sh sglang rocm /sgl-workspace/sglang

set -euo pipefail

usage() {
    echo "usage: $0 <repo> <platform> <target-dir>" >&2
    exit 2
}

[[ $# -eq 3 ]] || usage
repo="$1"
platform="$2"
target="$3"

# Resolve the patches/ tree as the sibling of this script's parent dir.
# Works for both in-repo invocations (scripts/apply-patches.sh, patches/
# sibling to scripts/) and in-image invocations (the Dockerfile COPYs
# both scripts/ and patches/ under /tmp/).
script_dir="$(cd "$(dirname "$0")" && pwd)"
patches_root="$(cd "$script_dir/.." && pwd)/patches/$repo"
[[ -d "$patches_root" ]] || {
    echo "apply-patches: cannot find $patches_root" >&2
    exit 1
}

[[ -d "$target/.git" ]] || {
    echo "apply-patches: $target is not a git repo" >&2
    exit 1
}

# Multi-platform iff a base/ subdir exists; otherwise flat / mono-platform.
multi=false
if [[ -d "$patches_root/base" ]]; then
    multi=true
fi

# Sanity-check that the platform subdir is the only one (besides base).
# Mono-platform repos refuse mismatched platform args so a typo in the
# Dockerfile (apply-patches.sh aiter cuda ...) surfaces loudly.
if ! $multi; then
    if ! ls "$patches_root"/*.patch >/dev/null 2>&1; then
        echo "apply-patches: no patches in $patches_root" >&2
        exit 1
    fi
    # Mono-platform repos only have patches for one platform; the only
    # honest assertion is that the caller passed *some* platform name
    # the repo was built for. We don't have a manifest, so we trust the
    # caller for now and let git am surface a real failure if patches
    # don't apply.
else
    if [[ ! -d "$patches_root/$platform" ]]; then
        echo "apply-patches: no $platform/ subdir under $patches_root" >&2
        echo "(add patches/$repo/$platform/ if this platform needs patches)" >&2
        exit 1
    fi
fi

# Fixed identity so format-patch output stays byte-stable across rebuilds.
export GIT_AUTHOR_NAME="vllm-shim"
export GIT_AUTHOR_EMAIL="vllm-shim@local"
export GIT_AUTHOR_DATE="2026-01-01T00:00:00+00:00"
export GIT_COMMITTER_NAME="vllm-shim"
export GIT_COMMITTER_EMAIL="vllm-shim@local"
export GIT_COMMITTER_DATE="2026-01-01T00:00:00+00:00"

cd "$target"

# Step 1: pin upstream. If refs/vllm-shim/upstream already exists we
# trust it: that's the upstream commit the existing patched/* branches
# were built on top of, and we want to rebuild against the same anchor.
# Only on the first run (no existing ref) do we set it from HEAD.
#
# After sync-repos.sh bumps the upstream version, refs/vllm-shim/upstream
# is stale; the dev must `git update-ref -d refs/vllm-shim/upstream`
# before re-running apply-patches.sh so we pick up the new HEAD. This
# is intentional: an automatic reset would silently rebase the patches
# onto whichever commit HEAD happens to be at, masking real "upstream
# drift broke the patches" failures.
if ! git rev-parse --verify --quiet refs/vllm-shim/upstream >/dev/null; then
    git update-ref refs/vllm-shim/upstream "$(git rev-parse HEAD)"
fi

apply_into_branch() {
    local branch="$1" start="$2" dir="$3"
    # checkout -B creates or resets the branch in one step, even if it
    # was already checked out (a plain `branch -f` would fail with "used
    # by worktree"). The script owns the patched/* namespace so wiping
    # on every run is the intended behavior.
    git checkout -B "$branch" "$start" --quiet
    shopt -s nullglob
    local patch
    for patch in "$dir"/*.patch; do
        echo "  [$branch] git am $(basename "$patch")"
        # -c commit.gpgsign=false: the patched/* commits are local
        #   scaffolding for the patch tooling; they're not user history
        #   and never get pushed. Without this override, a global
        #   commit.gpgsign=true (common on dev boxes) bakes a fresh
        #   signature into every replay, so the resulting commit SHAs
        #   drift and rebuild output isn't byte-stable.
        git -c commit.gpgsign=false am --quiet "$patch"
    done
    shopt -u nullglob
}

# git am refuses to apply onto a dirty index, but some base images ship
# upstream checkouts with uncommitted working-tree edits as build-time
# fixups (lmsysorg/sglang-rocm carries an uncommitted bounds-check fix to
# csrc/kernels/mhc_kernels.cu in /sgl-workspace/aiter, for example).
# Stash those before applying, restore them on top of patched/<platform>
# afterward so the kernels still compile against the base image's fixed
# source. Stash pop can conflict if a future patch touches the same
# file; treat that as a real build failure and resolve by hand.
stashed=false
if ! git diff-index --quiet HEAD --; then
    git stash push --quiet --include-untracked \
        -m "vllm-shim apply-patches: preserve working-tree fixups"
    stashed=true
fi

if $multi; then
    apply_into_branch "patched/base" "refs/vllm-shim/upstream" "$patches_root/base"
    apply_into_branch "patched/$platform" "patched/base" "$patches_root/$platform"
else
    apply_into_branch "patched/$platform" "refs/vllm-shim/upstream" "$patches_root"
fi

if $stashed; then
    git stash pop --quiet
fi

echo "apply-patches: $repo done (HEAD now on patched/$platform)"
