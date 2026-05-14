#!/usr/bin/env bash
# Regenerate the patches/ tree from the patched/* branches the dev has
# been editing in repos/<repo>/. Counterpart to apply-patches.sh.
#
# Read-only on the working tree: uses git format-patch with explicit
# ref ranges, never checks out a branch. The dev can be sitting on any
# branch (or detached HEAD) and rebuild a single tier without disturbing
# their state.
#
# Usage:
#   rebuild-patches.sh <repo>            # rebuild tier matching current
#                                        # branch (errors if not on patched/*)
#   rebuild-patches.sh <repo> base       # explicit tier
#   rebuild-patches.sh <repo> rocm
#   rebuild-patches.sh <repo> cuda
#   rebuild-patches.sh <repo> --all      # every tier the layout supports
#
# Examples:
#   rebuild-patches.sh sglang            # rebuild whatever I'm on
#   rebuild-patches.sh sglang --all      # full refresh before commit
#   rebuild-patches.sh aiter             # mono-platform; only one tier

set -euo pipefail

usage() {
    echo "usage: $0 <repo> [tier|--all]" >&2
    echo "       tier: base | rocm | cuda | ..." >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
repo="$1"
mode="${2:-}"

script_dir="$(cd "$(dirname "$0")" && pwd)"
shim_root="$(cd "$script_dir/.." && pwd)"
patches_root="$shim_root/patches/$repo"
repo_dir="$shim_root/repos/$repo"

[[ -d "$repo_dir/.git" ]] || {
    echo "rebuild-patches: $repo_dir is not a git repo" >&2
    exit 1
}
[[ -d "$patches_root" ]] || {
    echo "rebuild-patches: $patches_root not found" >&2
    exit 1
}

multi=false
if [[ -d "$patches_root/base" ]]; then
    multi=true
fi

# Helper: write all commits in <range> to <out_dir>, wiping <out_dir>
# first. Format-patch numbering (0001-, 0002-) provides apply order;
# the slug after the number comes from the commit subject.
rebuild_tier() {
    local range="$1" out_dir="$2"
    mkdir -p "$out_dir"
    rm -f "$out_dir"/*.patch
    echo "rebuild-patches: $range -> $out_dir"
    git -C "$repo_dir" format-patch --no-signature "$range" -o "$out_dir" >/dev/null
}

# Tier dispatch.
rebuild_one() {
    local tier="$1"
    if [[ "$tier" == "base" ]]; then
        $multi || {
            echo "rebuild-patches: $repo is mono-platform; no base tier" >&2
            exit 1
        }
        rebuild_tier "refs/vllm-shim/upstream..patched/base" "$patches_root/base"
    else
        # Platform tier.
        if $multi; then
            rebuild_tier "patched/base..patched/$tier" "$patches_root/$tier"
        else
            # Mono-platform: flat layout, no platform subdir.
            rebuild_tier "refs/vllm-shim/upstream..patched/$tier" "$patches_root"
        fi
    fi
}

# Resolve "current branch" mode.
if [[ -z "$mode" ]]; then
    head_ref="$(git -C "$repo_dir" symbolic-ref --short HEAD 2>/dev/null || true)"
    case "$head_ref" in
        patched/*) mode="${head_ref#patched/}" ;;
        "")
            echo "rebuild-patches: detached HEAD; pass a tier explicitly" >&2
            exit 1
            ;;
        *)
            echo "rebuild-patches: HEAD is on '$head_ref', not patched/*" >&2
            echo "(pass a tier explicitly, e.g.  $0 $repo rocm)" >&2
            exit 1
            ;;
    esac
fi

if [[ "$mode" == "--all" ]]; then
    if $multi; then
        rebuild_one base
    fi
    # Discover every platform-tier branch under refs/heads/patched/ that
    # isn't "base" and rebuild it. This auto-handles cuda once it lands
    # alongside rocm without script changes.
    while read -r ref; do
        tier="${ref#refs/heads/patched/}"
        [[ "$tier" == "base" ]] && continue
        rebuild_one "$tier"
    done < <(git -C "$repo_dir" for-each-ref --format='%(refname)' refs/heads/patched/)
else
    rebuild_one "$mode"
fi

echo "rebuild-patches: done"
