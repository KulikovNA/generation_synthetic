#stereo/FrameSeries.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class FrameGrid:
    """
    Описание сетки, в которой лежат кадры серии.

    Примеры:
      name="IR_LEFT"        -> оригинальная сетка IR_LEFT (как рендер)
      name="IR_LEFT_RECT"   -> сетка IR_LEFT после stereoRectify (виртуальная камера 1)
      name="COLOR"          -> сетка RGB
    """
    name: str
    rectify_meta: Optional[Any] = None  # обычно RectifyMaps/RectifyMeta; Any чтобы не ловить циклы импорта


class FrameSeries(list):
    """
    Обычный list кадров, но с полями grid / rectify_meta.

    Работает везде, где ожидается list:
      - len(...)
      - zip(...)
      - индексирование
      - итерации
    """
    def __init__(self, iterable: Iterable = (), *, grid: FrameGrid):
        super().__init__(iterable)
        self.grid = grid

    @property
    def rectify_meta(self):
        return self.grid.rectify_meta

    @property
    def grid_name(self) -> str:
        return self.grid.name