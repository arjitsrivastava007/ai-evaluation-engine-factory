"""Run the API with uvicorn."""

import os

import uvicorn

from eval_factory.api.app import create_app

app = create_app()


def main() -> None:
    uvicorn.run(
        "eval_factory.api.app:create_app",
        factory=True,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
