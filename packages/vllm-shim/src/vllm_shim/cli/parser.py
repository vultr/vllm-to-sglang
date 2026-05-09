from collections.abc import Sequence

from vllm_shim.values.parsed_args import ParsedArgs


class ArgParser:
    """Parses vLLM-style argv. Extracts model/host/port for the supervisor;
    everything else flows into ParsedArgs.passthrough for the backend's
    ArgTranslator to handle."""

    def parse(self, argv: Sequence[str]) -> ParsedArgs:
        model: str | None = None
        host = "0.0.0.0"
        port = 8000
        passthrough: list[str] = []

        i = 0
        args = list(argv)
        while i < len(args):
            arg = args[i]

            if arg == "serve":
                i += 1
                continue

            if arg == "--host" and i + 1 < len(args):
                host = args[i + 1]
                i += 2
                continue
            if arg.startswith("--host="):
                host = arg.split("=", 1)[1]
                i += 1
                continue

            if arg == "--port" and i + 1 < len(args):
                port = int(args[i + 1])
                i += 2
                continue
            if arg.startswith("--port="):
                port = int(arg.split("=", 1)[1])
                i += 1
                continue

            if arg in ("--model", "--model-name") and i + 1 < len(args):
                model = args[i + 1]
                i += 2
                continue
            if arg.startswith(("--model=", "--model-name=")):
                model = arg.split("=", 1)[1]
                i += 1
                continue

            if not arg.startswith("-") and model is None:
                model = arg
                i += 1
                continue

            passthrough.append(arg)
            i += 1

        if model is None:
            raise ValueError("No model specified in vLLM args")
        return ParsedArgs(model=model, host=host, port=port, passthrough=tuple(passthrough))
