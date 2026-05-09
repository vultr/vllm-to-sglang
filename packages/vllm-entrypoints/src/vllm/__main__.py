"""Stub: redirects `python -m vllm` to the vllm-shim entry point."""

from vllm_shim.cli.entrypoint import main

raise SystemExit(main())
