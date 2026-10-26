"""omni-lyra CLI entry point."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="omni-lyra",
        description="OMNI-Lyra: autonomous research agent with Lyra safety guardrails",
    )
    parser.add_argument("domain", help="Research domain description or path to .txt file")
    parser.add_argument("constraints", help="Path to YAML constraint file")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--image", default="python:3.12-slim", help="Docker image for sandbox")
    parser.add_argument("--model", default="nemotron-mini", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434")

    args = parser.parse_args()

    domain_path = Path(args.domain)
    domain_context = domain_path.read_text() if domain_path.exists() else args.domain

    from .loop import run_session
    asyncio.run(
        run_session(
            domain_context=domain_context,
            constraint_yaml=Path(args.constraints),
            max_steps=args.max_steps,
            sandbox_image=args.image,
            ollama_model=args.model,
            ollama_url=args.ollama_url,
        )
    )


if __name__ == "__main__":
    main()
