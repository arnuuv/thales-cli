"""
Maritime Generator - Fictitious Maritime Data Generator for Thales Mapping

This module generates 100% synthetic maritime data including:
- Restricted zones (MPA, military, windfarm)
- Vessel routes (ferry, TSS, irregular)
- Synthetic FLIR (infrared) vessel images

All data is procedurally generated - NO real-world maritime data is used.
"""

from .generator import MaritimeGenerator

__all__ = ['MaritimeGenerator']
__version__ = '1.0.0'
