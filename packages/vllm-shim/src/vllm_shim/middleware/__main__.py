"""Entry point for `python -m vllm_shim.middleware`; spawned by the supervisor."""

from vllm_shim.middleware.app import run

run()
