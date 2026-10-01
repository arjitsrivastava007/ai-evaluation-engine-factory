"""Errors the API layer can turn into HTTP responses."""


class FactoryError(Exception):
    """A user-facing failure with an HTTP status code."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
