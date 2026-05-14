# Upstream patches

The shim carries local patches against AITER and SGLang. They fix
bugs, enable workarounds for arch-specific kernel gaps, and add small
features the upstream maintainers haven't accepted (or haven't been
asked to accept). This doc covers how the patch tooling works and the
day-to-day workflow for adding or editing a patch.

## Why this exists

The shim runs on top of two upstream Python codebases (`aiter`,
`sglang`) that the Docker base image provides as git checkouts at a
pinned commit. We can't always wait on upstream for fixes:

- AMD kernels are arch-gated in C++; a missing `gfx94` enablement
  means we ship a Triton fallback locally.
- Quantization checkpoint quirks (Quark MXFP4, MoRI EP class names)
  need adapter code that's too narrow to upstream.
- Performance knobs (hipBLASLt heuristic API caps) are off in
  upstream but would slow down our tuner significantly without
  them.

Patches live in `patches/<repo>/...` at the repo root, replayed onto
the upstream checkout at image-build time as a stack of commits on
`patched/<platform>` branches. The dev workflow uses the same
machinery so the round-trip (edit a checkout, regenerate patches) is
clean.

## Layout

```
patches/
├── aiter/                       # mono-platform repo, flat layout
│   └── 0001-perf-gradlib-bound-hipBLASLt-…patch
└── sglang/                      # multi-platform repo, tiered layout
    ├── base/                    # applies on every Dockerfile.<platform>
    │   ├── 0001-fix-compressed-tensors-…patch
    │   ├── 0002-feat-deepseek-…patch
    │   └── 0003-fix-quark-exclude-…patch
    └── rocm/                    # only when building Dockerfile.rocm
        ├── 0001-feat-ep_moe-…patch
        └── 0002-feat-quark-mxfp4-triton-…patch
```

Rules:

- **Mono-platform repos** (AITER doesn't exist on CUDA) sit flat under
  `patches/<repo>/`. No `base/` or `<platform>/` subdir.
- **Multi-platform repos** (SGLang runs on both CUDA and ROCm) split
  into `base/` (always applied) and `<platform>/` (rocm, cuda, ...).
  The script applies `base/` first, then the matching platform tier
  on top.
- **Filename prefix `NNNN-`** is the apply order. `git format-patch`
  generates it from commit order on the patched branch; the rest of
  the filename comes from the commit subject after sanitization.

Classification cheat sheet for a new patch:

| Touches | Goes in |
|---|---|
| Pure Python, no `aiter.*` / HIP / CUDA imports | `sglang/base/` |
| Imports from `aiter.*` or uses MoRI / HIP-specific paths | `sglang/rocm/` |
| Imports from CUDA-only kernels / flashinfer / NCCL specifics | `sglang/cuda/` (create the dir when first needed) |
| AITER itself | `aiter/` (mono-platform) |

If a patch's runtime behavior is arch-gated (e.g. `gfx942` only), gate
it in the patched code via `is_gfx95_supported()` or similar; the
patches/ tree only encodes the *platform* axis, not the arch axis.

## Branch model

The scripts establish three named refs in each `repos/<repo>/`
checkout:

```
refs/vllm-shim/upstream                ← the upstream-pinned commit
    │
    ├── patched/base                   ← base/*.patch applied as commits
    │     │
    │     └── patched/<platform>       ← <platform>/*.patch on top
    │
    └── patched/<platform>             ← (mono-platform repos branch directly)
```

Each `.patch` is one commit on the corresponding branch. The commit's
subject is the patch's headline; the body is the rationale. `git am`
replays them; `git format-patch` regenerates them.

`refs/vllm-shim/upstream` is a custom ref (not under `refs/tags/` or
`refs/heads/`) so it doesn't collide with branch/tag operations and
sidesteps global `tag.forcesignannotated` configs. It's resolvable as
`vllm-shim/upstream` via `git rev-parse`'s DWIM rules.

## The scripts

Both live in `scripts/` and are plain bash with `set -euo pipefail`.

### `apply-patches.sh <repo> <platform> <target-dir>`

Replays patches onto a fresh checkout. Used by:
- the Dockerfile at image-build time
- the dev when first setting up `repos/<repo>/` for editing

Sequence:
1. If `refs/vllm-shim/upstream` doesn't yet exist, set it to current
   HEAD. (Already exists? Trust it; see "Gotchas" for what happens
   after `sync-repos.sh` bumps the upstream version.)
2. `git checkout -B patched/base refs/vllm-shim/upstream` (for
   multi-platform repos), then `git am base/*.patch`.
3. `git checkout -B patched/<platform> patched/base`, then
   `git am <platform>/*.patch`.
4. For mono-platform repos: steps 2 is skipped, step 3 roots
   `patched/<platform>` directly on `refs/vllm-shim/upstream`.

The script uses a fixed author identity and a fixed `GIT_AUTHOR_DATE`
(2026-01-01) and bypasses `commit.gpgsign` so the replayed commit
SHAs are byte-stable across rebuilds. `rebuild-patches.sh` depends on
this for idempotency.

### `rebuild-patches.sh <repo> [tier|--all]`

Regenerates `patches/<repo>/` from the current state of the
`patched/*` branches. Used by the dev after editing.

Modes:
- `rebuild-patches.sh <repo>`: derives the tier from the current
  branch (`patched/rocm` → `rocm`, `patched/base` → `base`). Errors
  if HEAD is detached or on a non-`patched/*` branch.
- `rebuild-patches.sh <repo> base|rocm|cuda`: explicit tier. The dev
  can be sitting anywhere; the script reads ref ranges, never checks
  out a branch.
- `rebuild-patches.sh <repo> --all`: rebuild every tier the layout
  supports. Auto-discovers platform tiers from `refs/heads/patched/*`.

Per-tier ranges:

| Tier | Range |
|---|---|
| `base` (multi-platform repo) | `refs/vllm-shim/upstream..patched/base` |
| `<platform>` on multi-platform repo | `patched/base..patched/<platform>` |
| `<platform>` on mono-platform repo | `refs/vllm-shim/upstream..patched/<platform>` |

Each pass clears the target directory then runs `git format-patch
--no-signature <range> -o <dir>`. Read-only on the working tree: no
checkout, no branch switch, the dev's in-flight work isn't disturbed.

## Workflows

### Adding a new patch

```bash
# 1. Make sure the patched branches are set up. Idempotent if they
#    already exist at the right place; first-time setup if not.
scripts/apply-patches.sh sglang rocm repos/sglang

# 2. Check out the tier you want to add to.
cd repos/sglang
git checkout patched/rocm        # ...or patched/base for cross-platform fixes

# 3. Edit files, then commit with a meaningful subject. The subject
#    becomes the patch's filename slug (after format-patch sanitizes
#    it). Project convention: conventional-commits style.
$EDITOR python/sglang/srt/...
git add -p
git -c commit.gpgsign=false commit -m "fix(scope): short description

Why this is needed (the problem).
How this fixes it (the implementation note).

Adapted from <source>; not upstreamed."

# 4. Regenerate the patches/ tree.
cd ../..
scripts/rebuild-patches.sh sglang
```

The new patch appears under `patches/sglang/rocm/NNNN-fix-scope-…patch`
with `NNNN` reflecting its position in the commit stack.

If you forget `-c commit.gpgsign=false` and your global config signs
commits, the resulting commit SHA is non-deterministic and rebuilds
won't be byte-stable. Recover by `git commit --amend
-c commit.gpgsign=false`.

### Editing an existing patch

```bash
# 1. Apply (if not already).
scripts/apply-patches.sh sglang rocm repos/sglang

# 2. Find the commit and edit it. Easiest is an interactive rebase
#    with the commit marked `edit` (or `reword` if you're only
#    changing the message).
cd repos/sglang
git rebase -i refs/vllm-shim/upstream patched/rocm
#   → mark the commit `edit`, save, hack on files, then:
git add -p
git -c commit.gpgsign=false commit --amend --no-edit
git rebase --continue

# 3. Regenerate.
cd ../..
scripts/rebuild-patches.sh sglang
```

For pure message edits, `git commit --amend` is fine; for content
edits, prefer rebase so you don't accidentally entangle two patches.

### Moving a patch between tiers

Cherry-pick across branches, then rebuild:

```bash
cd repos/sglang
git checkout patched/base
git cherry-pick <sha-from-rocm>
git checkout patched/rocm
git rebase --onto patched/base patched/base~1 patched/rocm  # drop the now-duplicated commit
cd ../..
scripts/rebuild-patches.sh sglang --all
```

`--all` is the right mode here because both tiers' filenames will
re-number.

### Bumping the upstream pin

When `scripts/sync-repos.sh::SGLANG_VERSION` (or AITER, etc.) moves to
a newer tag and you re-run sync-repos.sh, the local checkout's HEAD
advances but `refs/vllm-shim/upstream` and the `patched/*` branches
stay at the old commit. They're now stale.

```bash
# 1. Bump the version in scripts/sync-repos.sh and re-sync.
scripts/sync-repos.sh sglang

# 2. Drop the stale anchor so apply-patches.sh re-derives it from HEAD.
git -C repos/sglang update-ref -d refs/vllm-shim/upstream
git -C repos/sglang branch -D patched/base patched/rocm

# 3. Re-apply. If any patch fails, git am will stop and tell you which
#    one. Resolve, `git am --continue`, regenerate.
scripts/apply-patches.sh sglang rocm repos/sglang
scripts/rebuild-patches.sh sglang --all
```

The explicit "drop the ref before reapplying" step is intentional:
without it, `apply-patches.sh` keeps the old anchor and silently
rebases the patches forward (or fails inscrutably). Requiring the
drop makes upstream-drift breakage surface as `git am` failures with
the offending file in `repos/<repo>/` ready to inspect.

### What happens at image build

`docker/sglang/Dockerfile.rocm` runs:

```dockerfile
COPY patches/ /tmp/patches/
COPY scripts/ /tmp/scripts/
RUN bash /tmp/scripts/apply-patches.sh aiter  rocm /sgl-workspace/aiter  && \
    bash /tmp/scripts/apply-patches.sh sglang rocm /sgl-workspace/sglang && \
    mkdir -p /etc/vllm-shim && \
    git -C /sgl-workspace/aiter rev-parse --short=12 HEAD \
        > /etc/vllm-shim/aiter-cache-key && \
    rm -rf /tmp/patches /tmp/scripts
```

If any patch fails to apply, `git am` errors and the RUN step fails;
the image build aborts rather than silently producing a half-patched
image. The AITER cache-key write happens AFTER patching, so the SHA
captures upstream+patches together (see `docs/aiter.md`,
"JIT cache namespacing by AITER commit").

The Dockerfile.cuda doesn't currently call `apply-patches.sh` because
no CUDA-only or cross-platform patches need it. When the first such
patch lands under `patches/sglang/base/` or `patches/sglang/cuda/`,
add the matching `RUN bash /tmp/scripts/apply-patches.sh sglang cuda
/sgl-workspace/sglang` line.

## Conventions

### Commit messages

Conventional commits style: `<type>(<scope>): <subject>`.

- `<type>`: `feat` for new behavior, `fix` for bugs, `perf` for perf
  patches, `refactor` for restructuring without behavior change.
- `<scope>`: the file or component the patch lives in
  (`compressed-tensors`, `quark`, `ep_moe`, `gradlib`).
- `<subject>`: short imperative description. Lowercase. No trailing
  period.

Body should answer "why is this needed?" and "what was the design
choice?" The body becomes the explanatory prose at the top of the
`.patch` file; `git log --format=%B` on the patched branch shows it
too. Detailed bodies aren't padding: they're how a future maintainer
(or you, in six months) figures out whether the patch is still
needed.

End with a one-line provenance: `Adapted from <prior work>; not
upstreamed.` or `Upstream PR: <url>` if applicable.

### Why patches don't get pushed

The `patched/*` branches in `repos/<repo>/` are local scaffolding for
the patch tooling. They are not commits we author for the upstream
project; we don't sign them, we don't push them, and the script
bypasses `commit.gpgsign` precisely so they stay byte-stable
machine-generated artifacts. Treat them like build output, not like
human history.

The patches themselves are what we maintain; the branches are how the
scripts manipulate them.

## Gotchas

### Stale `refs/vllm-shim/upstream` after sync

After `scripts/sync-repos.sh` advances the upstream pin,
`refs/vllm-shim/upstream` points at the *old* commit. `apply-patches.sh`
will dutifully rebuild the `patched/*` branches on top of the old
commit and your patches won't reflect the new upstream at all.
Symptom: builds keep succeeding but you're not testing against the
version you bumped to.

Fix: drop the ref before re-applying.

```bash
git -C repos/sglang update-ref -d refs/vllm-shim/upstream
git -C repos/sglang branch -D patched/base patched/rocm
scripts/apply-patches.sh sglang rocm repos/sglang
```

### `commit.gpgsign=true` globally

If your dev box has `commit.gpgsign=true` in `~/.gitconfig`, every
commit gets a fresh signature with a timestamp; the resulting commit
SHA changes on every replay, and `rebuild-patches.sh` produces
different bytes each time (different `From <SHA>` line in the
mbox header).

`apply-patches.sh` passes `-c commit.gpgsign=false` to the `git am`
calls to suppress this. If you create commits *manually* on a
`patched/*` branch (not through `git am`), do the same:

```bash
git -c commit.gpgsign=false commit -m "..."
```

### `git am` fails midway

If a patch in the stack doesn't apply, `git am` stops and leaves the
working tree mid-apply. The CLI hint tells you what to do:

```
hint: When you have resolved this problem, run "git am --continue".
hint: If you prefer to skip this patch, run "git am --skip" instead.
hint: To restore the original branch and stop patching, run "git am --abort".
```

Typical recovery:

```bash
cd repos/sglang
$EDITOR <file>            # resolve the conflict by hand
git add <file>
git -c commit.gpgsign=false am --continue
# ...or abort and investigate:
git am --abort
```

After resolving, regenerate patches from the new state with
`rebuild-patches.sh`.

### Patch filename drift

When you reword a commit message, the resulting patch filename
changes (the `NNNN-<sanitized-subject>.patch` slug rebuilds from the
new subject). `rebuild-patches.sh` clears the target directory before
writing, so old-named files are dropped. If you forgot to commit
before rewording, your previous filename ends up deleted in
`git status`; check the diff to make sure the new file is the one you
expect.

### "Branch X used by worktree" errors

If you try `git branch -D patched/rocm` while it's the checked-out
branch, git refuses. Either:
- Check out a different branch first (`git checkout
  refs/vllm-shim/upstream` puts you on a detached HEAD pointing at
  the upstream commit).
- Or let `apply-patches.sh` handle it: `git checkout -B
  patched/rocm <start>` resets the branch even if it's currently
  checked out.

## See also

- `scripts/apply-patches.sh`, `scripts/rebuild-patches.sh`: the
  scripts themselves; comments at the top reiterate the contract.
- `scripts/sync-repos.sh`: how `repos/<repo>/` checkouts are
  populated and pinned.
- `docs/aiter.md`, section "Upstream patches": the AITER patch in
  prose form, and the cache-key namespacing that depends on
  HEAD-after-patching being stable.
- `docs/build-and-deploy.md`: how `apply-patches.sh` plugs into the
  Dockerfile and how the image-build matrix works.
