from pathlib import Path

from vllm_shim.cli.haproxy import HAProxyConfig
from vllm_shim.values.service_address import ServiceAddress


def test_render_includes_listen_bind() -> None:
    cfg = HAProxyConfig(
        listen=ServiceAddress("0.0.0.0", 8000),
        upstream=ServiceAddress("127.0.0.1", 8002),
    )
    text = cfg.render()
    assert "bind 0.0.0.0:8000" in text


def test_render_includes_upstream_server() -> None:
    cfg = HAProxyConfig(
        listen=ServiceAddress("0.0.0.0", 8000),
        upstream=ServiceAddress("127.0.0.1", 8002),
    )
    text = cfg.render()
    assert "server s1 127.0.0.1:8002" in text


def test_render_has_health_check_and_503_errorfile() -> None:
    cfg = HAProxyConfig(
        listen=ServiceAddress("h", 1),
        upstream=ServiceAddress("u", 2),
    )
    text = cfg.render()
    assert "option httpchk GET /health" in text
    assert "errorfile 503" in text


def test_write_to_creates_file(tmp_path: Path) -> None:
    cfg = HAProxyConfig(
        listen=ServiceAddress("h", 1),
        upstream=ServiceAddress("u", 2),
    )
    target = tmp_path / "haproxy.cfg"
    cfg.write_to(target)
    assert target.read_text() == cfg.render()
