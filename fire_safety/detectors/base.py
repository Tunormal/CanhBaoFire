from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..core.types import Detection, FrameContext


class BaseDetector(ABC):
    """Moi detector (rule hay model) deu theo interface nay -> de thay/ghep ve sau."""
    name = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray, ctx: FrameContext) -> List[Detection]:
        ...

    def reset(self) -> None:
        """Goi khi bat dau video moi."""
