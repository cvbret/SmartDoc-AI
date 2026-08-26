"""Backward-compatible access to a module logger.

Logging configuration is owned by ``app.core.logging``. This module no
longer installs its own handler so importing it cannot duplicate log output.
"""

import logging


logger = logging.getLogger(__name__)
