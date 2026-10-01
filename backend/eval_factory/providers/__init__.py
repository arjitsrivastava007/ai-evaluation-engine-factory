"""Provider wiring."""

from eval_factory.providers.llm import HeuristicProvider, build_provider

__all__ = ["HeuristicProvider", "build_provider"]
