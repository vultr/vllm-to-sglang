"""haproxy frontend: config templating, static error file, launch helper."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from vllm_shim.values.service_address import ServiceAddress

ERROR_503_BODY = (
    "HTTP/1.0 503 Service Unavailable\r\n"
    "Content-Length: 16\r\n"
    "Connection: close\r\n"
    "Content-Type: text/plain\r\n\r\n"
    "SGLang not ready"
)

ERROR_503_PATH = "/tmp/haproxy-errors/503-sglang.http"

_TEMPLATE = """global
  maxconn 4096

defaults
  mode http
  timeout connect 5s
  timeout client 300s
  timeout server 300s

frontend proxy
  bind {listen_host}:{listen_port}

  acl is_health path /health
  acl sglang_up nbsrv(sglang) gt 0
  http-request deny deny_status 200 if is_health sglang_up
  http-request deny deny_status 503 if is_health
  errorfile 503 {error_path}

  default_backend sglang

backend sglang
  option httpchk GET /health
  http-check expect status 200
  server s1 {upstream_host}:{upstream_port} check inter 5s fall 3 rise 2
"""


@dataclass(frozen=True, slots=True)
class HAProxyConfig:
    """haproxy config inputs: the public listen address and the upstream middleware address."""

    listen: ServiceAddress
    upstream: ServiceAddress

    def render(self) -> str:
        """Render the templated haproxy config to a string."""
        return _TEMPLATE.format(
            listen_host=self.listen.host,
            listen_port=self.listen.port,
            upstream_host=self.upstream.host,
            upstream_port=self.upstream.port,
            error_path=ERROR_503_PATH,
        )

    def write_to(self, path: Path) -> None:
        """Render and write the config to disk."""
        path.write_text(self.render())


def write_error_file() -> None:
    """Drops the static 503 errorfile haproxy serves when SGLang is down."""
    path = Path(ERROR_503_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ERROR_503_BODY)


def launch(config_path: Path) -> subprocess.Popen[bytes]:
    """Spawn haproxy with the given config file; caller is responsible for the process."""
    return subprocess.Popen(["haproxy", "-f", str(config_path)])
