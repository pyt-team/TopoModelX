# Copyright © 2022 Pyt-Team
# All rights reserved.

"""
Base classes for TorchTopo exceptions
"""


class TorchTopoException(Exception):
    """Base class for exceptions in TorchTopo."""


class TorchTopoError(TorchTopoException):
    """Exception for a serious error in TorchTopo"""


class TorchTopoNotImplementedError(TorchTopoError):
    """Exception for methods not implemented for an object type."""
