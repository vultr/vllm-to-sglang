# Build and deploy

The shim ships as container images, one per `(backend, platform)` combination. The build is structured around a per-backend matrix that Jenkins discovers automatically.

## Image matrix

The directory layout under `docker/` *is* the matrix:

```
docker/
└── <backend>/
    └── Dockerfile.<platform>
```

Today:

```
docker/sglang/Dockerfile.cuda
docker/sglang/Dockerfile.rocm
docker/trtllm/Dockerfile.cuda
docker/vllm/Dockerfile.cuda
docker/vllm/Dockerfile.rocm
```

Adding a backend or platform means adding a Dockerfile at the right path. Jenkins does the rest.

## Base image and AITER pin policy

The base image tags in each `docker/<backend>/Dockerfile.<platform>` (and the `AITER_VERSION` pin in `scripts/sync-repos.sh`) are the only knobs that decide what upstream code the shim runs against. Picking those values is a judgement call, but a few rules are non-negotiable; bumping pins without checking them has cost us debugging time before.

### SGLang base image: prefer un-dated tags

lmsysorg publishes both clean release tags and date-suffixed snapshot tags. The two carry **different commits** for the same labeled version:

| Tag pattern | What ships | Use? |
|---|---|---|
| `lmsysorg/sglang:vX.Y.Z-cu130` | git HEAD = upstream `vX.Y.Z` tag (clean release build) | Yes |
| `lmsysorg/sglang:vX.Y.Z-rocm720-mi30x` | git HEAD = upstream `vX.Y.Z` tag (clean release build) | Yes |
| `lmsysorg/sglang-rocm:vX.Y.Z-rocm720-mi30x-YYYYMMDD` | git HEAD is upstream `vX.Y.Z` + N commits as of YYYYMMDD; "vX.Y.Z" is a baseline marker, not the actual commit | No |
| `lmsysorg/sglang:nightly-dev-cuNN-YYYYMMDD-SHORTSHA` | git HEAD = the snapshotted commit; the date+SHA suffix is the real identifier | No (unless deliberately tracking nightlies) |

Verification before pinning a new base image:

```bash
docker run --rm --entrypoint bash <image> -c \
  'git -C /sgl-workspace/sglang describe --tags --always && \
   git -C /sgl-workspace/sglang rev-parse HEAD'
```

`describe` should print exactly the labeled version (`v0.5.12`), not `vX.Y.Z-N-gSHORT`. A trailing `-N-gSHORT` means the image is N commits past the tag and the label is misleading.

### vLLM base image: enforce CUDA/ROCm commit parity

`vllm/vllm-openai:vX.Y.Z-cuNNN` and `vllm/vllm-openai-rocm:vX.Y.Z` ship as `pip install` wheels (no in-image git checkout), but the upstream commit is stamped in the wheel's `_version.py`. A single `VLLM_VERSION` pin in `sync-repos.sh` only makes sense as long as both images report the **same** `__commit_id__`. AMD's ROCm fork has historically tracked upstream tags closely; if it ever lags or leads, this stops being true and the shim needs per-platform pinning (or to pick one and accept divergence in `repos/vllm/`).

Verification before pinning a new vLLM tag:

```bash
for img in vllm/vllm-openai:vX.Y.Z-cuNNN vllm/vllm-openai-rocm:vX.Y.Z; do
  echo "=== $img ==="
  docker run --rm --entrypoint bash $img -c \
    'python3 -c "from vllm._version import __commit_id__; print(__commit_id__)"'
done
```

Both must print the same `gSHORT` value. Resolve the full 40-char SHA via `git ls-remote https://github.com/vllm-project/vllm.git vX.Y.Z` and that's the `VLLM_VERSION` pin.

### AITER: pin to whichever published version is higher

The two ROCm base images both ship AITER but in different forms: `lmsysorg/sglang:*-rocm720-mi30x` carries an editable git checkout at `/sgl-workspace/aiter`, and `vllm/vllm-openai-rocm:*` carries the `amd-aiter` pip wheel. Their AITER versions usually differ. The policy is to pin `AITER_VERSION` in `sync-repos.sh` to **whichever PEP440 version is higher**, then reset both images to that commit at build time (`docker/sglang/Dockerfile.rocm` does a `git fetch + reset --hard`; `docker/vllm/Dockerfile.rocm` does a `git init` bootstrap into `/opt/aiter` and a source rebuild). Both images then converge on the same patched AITER tree and share the runtime JIT cache namespace cleanly.

Note that `vX.Y.Z.postN+M` does **not** promote past `vX.Y.(Z+1)`: a mid-stream `v0.1.12.post1+150` commit ranks below the `v0.1.13` release tag, even though it's chronologically later.

Verification before bumping `AITER_VERSION`:

```bash
# AITER commit in the SGLang ROCm base image:
docker run --rm --entrypoint bash <sglang-rocm-base> -c \
  'git -C /sgl-workspace/aiter describe --tags --always && \
   git -C /sgl-workspace/aiter rev-parse HEAD'

# AITER wheel version in the vLLM ROCm base image:
docker run --rm --entrypoint bash <vllm-rocm-base> -c \
  'pip show amd-aiter | grep -i ^Version'
```

Compare the two versions in PEP440 order, pick the higher, resolve the corresponding commit upstream (`git ls-remote https://github.com/ROCm/aiter.git refs/tags/vX.Y.Z`), and that's the `AITER_VERSION` pin.

### Bump procedure (all repos)

After editing pins in `scripts/sync-repos.sh` and/or the base-image `ARG` in a Dockerfile:

```bash
# 1. Sync local checkouts (read-only on patches/).
scripts/sync-repos.sh <repo>

# 2. Drop the stale anchor so apply-patches.sh re-derives from the new HEAD.
git -C repos/<repo> update-ref -d refs/vllm-shim/upstream
git -C repos/<repo> branch -D patched/base patched/<tier1> patched/<tier2>...

# 3. Re-apply. Patches that conflict with the new upstream surface as
#    `git am` failures; resolve in repos/<repo>, then re-apply.
scripts/apply-patches.sh <repo> <tier> repos/<repo>

# 4. Regenerate patches/<repo>/ from any conflict resolutions, so the new
#    base/ patches are byte-stable.
scripts/rebuild-patches.sh <repo> --all

# 5. Build the affected images and verify.
docker compose -f docker/<backend>/compose.<platform>.yaml build
```

See `docs/patches.md` for the conflict-resolution loop in step 3 and the byte-stability rules around `commit.gpgsign`.

## Dockerfile shape

The simplest case is `docker/sglang/Dockerfile.cuda`, which is also the canonical pattern:

```dockerfile
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.12-cu130

FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends haproxy && \
    rm -rf /var/lib/apt/lists/*

# Apply local patches to the upstream checkout in /sgl-workspace.
COPY patches/ /tmp/patches/
COPY scripts/ /tmp/scripts/
RUN bash /tmp/scripts/apply-patches.sh sglang cuda /sgl-workspace/sglang && \
    rm -rf /tmp/patches /tmp/scripts

# Fresh transformers from upstream main, past PyPI, so newer model
# architectures land without waiting on a release cycle.
RUN uv pip install --no-cache \
        --python /usr/bin/python \
        --python-preference only-system \
        --break-system-packages \
        --upgrade git+https://github.com/huggingface/transformers.git

COPY pyproject.toml uv.lock /src/
COPY packages /src/packages
RUN uv pip install --no-cache \
        --python /usr/bin/python \
        --python-preference only-system \
        --break-system-packages \
        /src/packages/vllm-shim \
        /src/packages/vllm-entrypoints \
        distro
```

Seven things going on:

1. **`ARG BASE_IMAGE`**: pinned at the top so it's overridable per build (`--build-arg BASE_IMAGE=…`) but has a sane default. The default tracks the upstream backend image we're known to work against.
2. **Multi-stage `COPY --from=ghcr.io/astral-sh/uv:…`**: copies `uv` and `uvx` from the official Astral image. Avoids a curl-and-install dance and keeps the version explicit.
3. **`apt-get install haproxy`**: the only system package we add. The base image already has CUDA/ROCm + Python + the inference engine.
4. **`apply-patches.sh` against the in-image upstream checkout.** `patches/` and `scripts/` are COPYed into `/tmp` so the script can replay every `.patch` under `patches/<repo>/[<tier>/]` onto the base image's git checkout at `/sgl-workspace/<repo>`. The install in step 6 picks up the patched source because the base image installs the upstream package editably from that checkout. `/tmp/patches` and `/tmp/scripts` are deleted in the same layer so they don't bloat the final image. See `docs/patches.md` for the layout and the dev round-trip.
5. **Bleeding-edge `transformers` upgrade.** Forces `transformers` from upstream main, past both the base image's pin and the latest PyPI release. New model architectures (recent Kimi/Qwen revisions, etc.) and their fixes land without waiting on a `transformers` release. Same `--python` / `--python-preference` / `--break-system-packages` triad as step 6.
6. **Source copy + targeted install.** `pyproject.toml`, `uv.lock`, and `packages/` go into `/src/`. The `--python /usr/bin/python` flag pins which interpreter receives the install (the SGLang/TRT-LLM CUDA images have other interpreters lying around; without this flag uv may pick the wrong one or download a managed Python). `--python-preference only-system` blocks the managed-Python fallback. `--break-system-packages` is needed because Ubuntu 24.04 marks the system site-packages as PEP 668 externally-managed; uv refuses without it.
7. **`distro`** (SGLang CUDA only): a required `openai` runtime dependency that the upstream `lmsysorg/sglang:vX.Y.Z-cu130` image bundles `openai` without. Without it, `import sglang.launch_server` fails. Added at the same install step.

### Per-image divergence

Convergence isn't a goal anymore: each base image has its own quirks the Dockerfile has to absorb.

**`docker/trtllm/Dockerfile.cuda`** is the closest cousin to the canonical example: same install structure, same upstream-`transformers` upgrade, no `distro` (the NVIDIA TRT-LLM image's `openai` install is complete), no `apply-patches.sh` block (there are no TRT-LLM patches today; the COPY+RUN pair will be added when the first patch lands), different `BASE_IMAGE` default (`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`).

**`docker/sglang/Dockerfile.rocm`** diverges more substantially because the AMD/ROCm base image's Python is 3.10 and vllm-shim requires 3.12:

```dockerfile
# AMD's aiter kernels belong with SGLang in its existing 3.10 venv.
RUN uv pip install --no-cache --python /opt/venv/bin/python /sgl-workspace/aiter

# vllm-shim lives in its own Python 3.12 venv so it can satisfy
# requires-python without disturbing SGLang's 3.10 environment.
RUN uv venv /opt/shim --python 3.12

COPY pyproject.toml uv.lock /src/
COPY packages /src/packages
RUN uv pip install --no-cache --python /opt/shim/bin/python \
        /src/packages/vllm-shim \
        /src/packages/vllm-entrypoints

ENV PATH="/opt/shim/bin:/opt/venv/bin:${PATH}"
```

Three things to note:

- **`aiter` lands in SGLang's 3.10 `/opt/venv`.** That's where SGLang lives and where it expects to find aiter at import time.
- **vllm-shim lives in a separate 3.12 venv at `/opt/shim`.** The launcher spawns SGLang via the `sglang` console script (resolved via PATH); the script's shebang routes execution into `/opt/venv/bin/python`, so the two interpreters never need to share a `sys.path`. See `docs/backends.md` for the launcher fallback that makes this work.
- **PATH ordering is load-bearing.** `/opt/shim/bin` first ensures `vllm` resolves to our entrypoint; `/opt/venv/bin` next ensures `sglang` resolves to upstream's wrapper.
- **Local patches against AITER and SGLang.** The Dockerfile.rocm also COPYs `patches/` and `scripts/` into the image and runs `scripts/apply-patches.sh aiter sglang /sgl-workspace/aiter` and `scripts/apply-patches.sh sglang rocm /sgl-workspace/sglang` *before* the `uv pip install` step, so the installed packages reflect our local fixes. A preceding `git fetch + reset --hard ${AITER_VERSION}` resets the bundled AITER checkout to the shim's pinned commit (driven by `scripts/sync-repos.sh::AITER_VERSION`), so this image and the vLLM ROCm image converge on the same AITER source. See `docs/patches.md` for the layout and the dev round-trip via `scripts/rebuild-patches.sh`.

**`docker/vllm/Dockerfile.cuda` and `docker/vllm/Dockerfile.rocm`** use the same `/opt/shim` venv isolation pattern as the SGLang ROCm image, but for a different reason. The `vllm-entrypoints` stub package is what intercepts `python -m vllm.X` invocations; if it shared an interpreter with real vLLM, the stub would shadow the real package and the launcher's `python -m vllm.entrypoints.openai.api_server` would bounce back into the shim. Installing vllm-shim and `vllm-entrypoints` into a separate `/opt/shim` 3.12 venv keeps the stub off real vLLM's `sys.path`. The launcher reaches real vLLM via `VLLM_BACKEND_PYTHON` (default `/usr/bin/python3`), pointing at the base image's interpreter where vLLM is installed. Both Dockerfiles also clear the base image's inherited `ENTRYPOINT ["vllm", "serve"]` (the shim's `vllm` console script is on PATH now, so leaving the inherited entrypoint in place would prepend a second `vllm serve` to every invocation).

The ROCm variant additionally bootstraps a real AITER git checkout at `/opt/aiter` (the base image ships AITER as the `amd-aiter` pip wheel with no git tree to patch). `pip uninstall amd-aiter` removes the wheel, `git init` + `git remote add` + `git fetch --depth 1 ${AITER_VERSION}` + `git checkout -f FETCH_HEAD` constructs a repo at the same commit `scripts/sync-repos.sh` pins, `scripts/apply-patches.sh aiter vllm /opt/aiter` applies the base + vllm tier patches, and `uv pip install /opt/aiter` rebuilds the compiled artifacts against the patched source. The Dockerfile then captures `git -C /opt/aiter rev-parse --short=12 HEAD` to `/etc/vllm-shim/aiter-cache-key`, mirroring the SGLang ROCm Dockerfile so both images participate in the same runtime JIT cache namespacing scheme (see `docs/aiter.md`, "JIT cache namespacing by AITER commit").

## Build context

Build from the repo root, not from inside `docker/`:

```bash
docker build -f docker/sglang/Dockerfile.rocm -t vllm-shim:sglang-rocm .
```

The trailing `.` is the build context. The Dockerfiles `COPY pyproject.toml uv.lock /src/` and `COPY packages /src/packages`; both paths are relative to the context, so building from anywhere else won't find them.

`.dockerignore` doesn't currently exist; if/when the repo grows large fixtures, add one to keep the context tight.

## Jenkins matrix

The pipeline is in `Jenkinsfile`. Three stages:

### Discover

```groovy
def lines = sh(returnStdout: true, script: 'ls docker/*/Dockerfile.* 2>/dev/null || true').trim()
def combos = lines ? lines.split('\n').collect { p ->
    def parts = p.split('/')
    [backend: parts[1], platform: parts[2] - 'Dockerfile.']
} : []
if (params.BACKEND?.trim())  { combos = combos.findAll { it.backend  == params.BACKEND.trim()  } }
if (params.PLATFORM?.trim()) { combos = combos.findAll { it.platform == params.PLATFORM.trim() } }
```

A shell `ls` walks `docker/*/Dockerfile.*` from the workspace root, then Groovy splits each result into `{backend, platform}`. (`findFiles` would be the obvious Groovy-native call but it pulls in the Pipeline Utility Steps plugin; the `ls` form avoids that dependency.) The `BACKEND` and `PLATFORM` job parameters narrow the matrix; both empty means "build everything found."

If the resulting set is empty, the pipeline errors out. This protects against silently building nothing when a typo leaves no matches.

### Build & push (parallel)

```groovy
def jobs = env.COMBOS.split(',').collectEntries { combo ->
    ...
    docker.withRegistry("https://${env.REGISTRY_HOST}", env.CRED_ID) {
        sh """
            docker buildx build \\
                --file ${dockerfile} \\
                --tag ${imageRef} \\
                --cache-from type=registry,ref=${cacheRef} \\
                --cache-to   type=registry,ref=${cacheRef},mode=min,ignore-error=true \\
                --push \\
                .
        """
    }
}
parallel jobs
```

Each `(backend, platform)` becomes a parallel branch. Each branch:

- Authenticates to the Vultr container registry using the `ATL_VCR_VLLM` Jenkins credential.
- Runs `docker buildx build` with registry-backed cache. `mode=min` stores only the final image's layers (not intermediate stages) and `ignore-error=true` keeps the build green if the cache push itself fails (e.g. transient registry hiccup). This trades cache hit rate on edits to intermediate layers for lower registry-storage cost and noisy-cache-write resilience; if rebuilds start feeling slow because intermediate layers are missing from cache, `mode=max` is the knob to flip.
- Tags the result `${TAG}-${BACKEND}-${PLATFORM}` and pushes.

### Tag scheme

| Variable | Default | Example |
|---|---|---|
| `TAG`      | `nightly` | `nightly`, `v1.2.3`, `pr-42` |
| `BACKEND`  | `''` (all) | `sglang` |
| `PLATFORM` | `''` (all) | `rocm`, `cuda` |

Final image tag: `atl.vultrcr.com/vllm/vllm-shim:${TAG}-${BACKEND}-${PLATFORM}`.

A separate cache tag `…-${BACKEND}-${PLATFORM}-cache` lives alongside it. Don't garbage-collect cache tags blindly; losing them means full rebuilds on every commit.

## Triggering builds

### From the CLI

```bash
curl -X POST "https://jenkins.sweetapi.com/job/vllm-shim/buildWithParameters" \
  -d TAG=nightly \
  -d BACKEND=sglang \
  -d PLATFORM=rocm
```

`BACKEND` and `PLATFORM` can be omitted to build the full matrix.

### From the Jenkins UI

The `Build with Parameters` form exposes `BRANCH`, `TAG`, `BACKEND`, `PLATFORM`. Defaults are sensible. For a smoke build of the latest master across both platforms, just hit `Build`.

## Local builds

For development the easiest path is the per-backend compose files under `docker/<backend>/`:

```bash
# CUDA + SGLang
docker compose -f docker/sglang/compose.cuda.yaml up

# CUDA + TRT-LLM
docker compose -f docker/trtllm/compose.cuda.yaml up

# ROCm + SGLang
docker compose -f docker/sglang/compose.rocm.yaml up

# CUDA + vLLM
docker compose -f docker/vllm/compose.cuda.yaml up

# ROCm + vLLM
docker compose -f docker/vllm/compose.rocm.yaml up
```

Each compose file builds the image, requests a GPU (NVIDIA via `deploy.resources`, AMD via `/dev/kfd` + `/dev/dri` device mounts and the `video`/`render` groups), mounts the host's HuggingFace cache at `/root/.cache/huggingface`, sets `VLLM_SHIM_BACKEND` correctly, and runs `vllm serve ${MODEL:-Qwen/Qwen2.5-0.5B-Instruct} --port 8000`. Override the model with `MODEL=...` in the environment.

The Qwen 0.5B default is small enough to smoke-test on a single consumer GPU (~1 GB weights). For larger models, set `MODEL` and add `--tensor-parallel-size`, `--max-model-len`, etc. via the compose `command:` field or by running directly:

```bash
docker build -f docker/sglang/Dockerfile.cuda -t vllm-shim:dev-cuda .
docker run --rm --gpus all -p 8000:8000 vllm-shim:dev-cuda \
    vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --tensor-parallel-size 1 --max-model-len 4096
```

Note that `vllm serve` is the public CLI; `vllm-shim` doesn't expose its own command. From the container's perspective, you're launching vLLM; the shim intercepts.

## Image size

The upstream base images are large (SGLang ROCm ~15 GB, SGLang CUDA ~10 GB, TRT-LLM CUDA ~20 GB). The shim adds:

- haproxy (~5 MB).
- uv (~30 MB).
- The two workspace packages (~50 KB of Python, plus FastAPI + httpx + uvicorn dependencies, ~20 MB total).
- ROCm only: a separate Python 3.12 venv at `/opt/shim` (~50 MB for the managed CPython tarball plus the shim's deps).

Total shim overhead: under 100 MB on top of whatever the base image weighs (closer to 150 MB on ROCm). The base image dominates by two orders of magnitude.

## Adding a new platform or backend

The pipeline is fully driven by the file layout. To add e.g. `docker/sglang/Dockerfile.tpu`:

1. Drop the Dockerfile in place.
2. Push to the branch Jenkins is tracking.
3. Run a build (or wait for the next scheduled one).

Jenkins will pick up the new combo automatically. No pipeline edits needed.

The same applies to a new backend: add `docker/<backend>/Dockerfile.<platform>` files at the right paths and the matrix grows. (Don't forget to wire the backend into the registry; see `docs/backends.md`.)
