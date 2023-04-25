"""Base classes for exceptions."""


class TorchTopoException(Exception):
    """Base class for exceptions."""


class TorchTopoError(TorchTopoException):
    """Exception for a serious error."""


class TorchTopoNotImplementedError(TorchTopoError):
    """Exception for methods not implemented for an object type."""
