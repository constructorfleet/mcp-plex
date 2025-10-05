from __future__ import annotations

from typing import List

from qdrant_client import models

StorageBatch = List[models.PointStruct]

__all__ = ["StorageBatch"]
