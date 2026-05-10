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
```

Adding a backend or platform means adding a Dockerfile at the right path. Jenkins does the rest.

## Dockerfile shape

The simplest case is `docker/sglang/Dockerfile.cuda`, which is also the canonical pattern:

```dockerfile
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.11-cu130-runtime

FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends haproxy && \
    rm -rf /var/lib/apt/lists/*

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

Five things going on:

1. **`ARG BASE_IMAGE`**: pinned at the top so it's overridable per build (`--build-arg BASE_IMAGE=…`) but has a sane default. The default tracks the upstream backend image we're known to work against.
2. **Multi-stage `COPY --from=ghcr.io/astral-sh/uv:…`**: copies `uv` and `uvx` from the official Astral image. Avoids a curl-and-install dance and keeps the version explicit.
3. **`apt-get install haproxy`**: the only system package we add. The base image already has CUDA/ROCm + Python + the inference engine.
4. **Source copy + targeted install.** `pyproject.toml`, `uv.lock`, and `packages/` go into `/src/`. The `--python /usr/bin/python` flag pins which interpreter receives the install (the SGLang/TRT-LLM CUDA images have other interpreters lying around; without this flag uv may pick the wrong one or download a managed Python). `--python-preference only-system` blocks the managed-Python fallback. `--break-system-packages` is needed because Ubuntu 24.04 marks the system site-packages as PEP 668 externally-managed; uv refuses without it.
5. **`distro`** (SGLang CUDA only): a required `openai` runtime dependency that the upstream `lmsysorg/sglang:cu130-runtime` image bundles `openai` without. Without it, `import sglang.launch_server` fails. Added at the same install step.

### Per-image divergence

Convergence isn't a goal anymore: each base image has its own quirks the Dockerfile has to absorb.

**`docker/trtllm/Dockerfile.cuda`** is the closest cousin to the canonical example: same install structure, no `distro` (the NVIDIA TRT-LLM image's `openai` install is complete), different `BASE_IMAGE` default (`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`).

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
def combos = findFiles(glob: 'docker/*/Dockerfile.*').collect { f ->
    [backend: f.path.split('/')[1], platform: f.name - 'Dockerfile.']
}
if (params.BACKEND?.trim())  { combos = combos.findAll { it.backend == params.BACKEND.trim() } }
if (params.PLATFORM?.trim()) { combos = combos.findAll { it.platform == params.PLATFORM.trim() } }
```

Walks `docker/*/Dockerfile.*`, parses the directory and suffix into `{backend, platform}` tuples. The `BACKEND` and `PLATFORM` job parameters narrow the matrix; both empty means "build everything found."

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
                --cache-to   type=registry,ref=${cacheRef},mode=max \\
                --push \\
                .
        """
    }
}
parallel jobs
```

Each `(backend, platform)` becomes a parallel branch. Each branch:

- Authenticates to the Vultr container registry using the `ATL_VCR_VLLM` Jenkins credential.
- Runs `docker buildx build` with registry-backed cache (`mode=max` so the cache itself stores intermediate layers, not just the final manifest).
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
