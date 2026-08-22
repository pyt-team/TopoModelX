"""Base classes for TopoModelX."""

from .aggregation import Aggregation
from .conv import Conv
from .message_passing import MessagePassing

__all__ = ["Aggregation", "Conv", "MessagePassing"]
