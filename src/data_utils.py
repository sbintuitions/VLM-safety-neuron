"""SafetyQuery: one query of the detection data (text, optional image, label)."""

from dataclasses import dataclass
from typing import Dict, Optional

from PIL import Image


@dataclass
class SafetyQuery:
    text: str
    image: Optional[Image.Image] = None
    label: str = "harmful"  # or "safe"
    harm_category: Optional[str] = None  # ViSafe-Detect harm category (harmful images)
    metadata: Optional[Dict] = None
