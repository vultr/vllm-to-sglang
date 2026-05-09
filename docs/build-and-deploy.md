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
```

Adding a backend or platform means adding a Dockerfile at the right path. Jenkins does the rest.

## Dockerfile shape

Each Dockerfile follows the same pattern. From `docker/sglang/Dockerfile.rocm`:

```dockerfile
ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.11-rocm720-mi30x-20260508

FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends haproxy && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /src/
COPY packages /src/packages
RUN uv pip install --system --no-cache \
        /src/packages/vllm-shim \
        /src/packages/vllm-entrypoints \
        /sgl-workspace/aiter
```

Five things going on:

1. **`ARG BASE_IMAGE`**: pinned at the top so it's overridable per build (`--build-arg BASE_IMAGE=…`) but has a sane default. The default tracks the upstream SGLang image we're known to work against.
2. **Multi-stage `COPY --from=ghcr.io/astral-sh/uv:…`**: copies `uv` and `uvx` from the official Astral image. Avoids a curl-and-install dance and keeps the version explicit.
3. **`apt-get install haproxy`**: the only system dependency. The base image already has CUDA/ROCm + Python + SGLang.
4. **Source copy.** `pyproject.toml`, `uv.lock`, and `packages/` are dropped into `/src/`. The two workspace packages get installed into the system Python.
5. **`aiter`** (ROCm only): installed from `/sgl-workspace/aiter` inside the SGLang ROCm image. AMD's attention kernel library; SGLang on ROCm needs it. CUDA's image has its equivalent baked in already.

The CUDA Dockerfile is identical except the `BASE_IMAGE` default and no `aiter` install. Convergence is intentional; diverging the two on anything other than the base image is a smell.

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

For development, build directly without the registry:

```bash
docker build -f docker/sglang/Dockerfile.cuda -t vllm-shim:dev-cuda .
docker run --rm --gpus all -p 8000:8000 vllm-shim:dev-cuda \
    vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --tensor-parallel-size 1 --max-model-len 4096
```

Note that `vllm serve` is the public CLI; `vllm-shim` doesn't expose its own command. From the container's perspective, you're launching vLLM; the shim intercepts.

## Image size

The base SGLang images are large (~15 GB for ROCm, ~10 GB for CUDA). The shim adds:

- haproxy (~5 MB).
- uv (~30 MB).
- The two workspace packages (~50 KB of Python, plus FastAPI + httpx + uvicorn dependencies, ~20 MB total).

Total shim overhead: under 100 MB on top of whatever the base image weighs. The base image dominates by two orders of magnitude.

## Adding a new platform or backend

The pipeline is fully driven by the file layout. To add e.g. `docker/sglang/Dockerfile.tpu`:

1. Drop the Dockerfile in place.
2. Push to the branch Jenkins is tracking.
3. Run a build (or wait for the next scheduled one).

Jenkins will pick up the new combo automatically. No pipeline edits needed.

The same applies to a new backend: add `docker/<backend>/Dockerfile.<platform>` files at the right paths and the matrix grows. (Don't forget to wire the backend into the registry; see `docs/backends.md`.)
