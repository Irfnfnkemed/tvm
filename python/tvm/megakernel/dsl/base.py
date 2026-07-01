from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

# ============================================================
# Low-level tile implementation
# ============================================================


class TileImpl(ABC):
    """
    Low-level tile implementation.

    This class only describes HOW a tile runs.
    It should not know the global megakernel graph.
    """

    @classmethod
    def class_init(cls, smem_manager):
        pass

    @classmethod
    def class_finalize(cls):
        pass

    @classmethod
    def class_name(cls):
        return cls.__name__

    def __init__(self):
        self._instance_id = id(self)

    def __str__(self):
        class_name = self.__class__.__name__
        return f"{class_name}-{self._instance_id:x}"

    def init(self, smem_manager):
        pass

    def host_init(self):
        pass

    @abstractmethod
    def run(self, m_idx, n_idx, k_idx):
        raise NotImplementedError("run is not implemented")


# ============================================================
# Tensor / Event / Dependency
# ============================================================


@dataclass
class TensorSpec:
    name: str
    shape: Any = None
    dtype: str | None = None


@dataclass
class EventSpec:
    name: str
    shape: Any
    init: Any
    dtype: str = "int32"


@dataclass
class DepSpec:
    event: EventSpec
    coord_map: Callable
    expected: Any | None = None


# ============================================================
# TileSpec
# ============================================================


@dataclass
class TileSpec:
    name: str
    impl: TileImpl
    tile_num: tuple
    attrs: dict[str, Any] = field(default_factory=dict)
    reads: list[TensorSpec] = field(default_factory=list)
    writes: list[TensorSpec] = field(default_factory=list)
    waits: list[DepSpec] = field(default_factory=list)
    notifies: list[DepSpec] = field(default_factory=list)

    def read(self, *tensors: TensorSpec):
        self.reads.extend(tensors)
        return self

    def write(self, *tensors: TensorSpec):
        self.writes.extend(tensors)
        return self

    def wait(
        self,
        event: EventSpec,
        coord_map: Callable,
        *,
        expected: Any,
    ):
        self.waits.append(
            DepSpec(
                event=event,
                coord_map=coord_map,
                expected=expected,
            )
        )
        return self

    def notify(
        self,
        event: EventSpec,
        coord_map: Callable,
    ):
        self.notifies.append(
            DepSpec(
                event=event,
                coord_map=coord_map,
            )
        )
        return self


# ============================================================
# KernelSpec
# ============================================================


class KernelSpec:
    def __init__(self, name: str):
        self.name = name
        self.tensors: dict[str, TensorSpec] = {}
        self.events: dict[str, EventSpec] = {}
        self.tiles: list[TileSpec] = []

    def tensor(self, name: str, shape=None, dtype=None):
        if name in self.tensors:
            raise ValueError(f"Duplicate tensor: {name}")

        t = TensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
        )
        self.tensors[name] = t
        return t

    def event(self, name: str, shape: Any, init: Any, dtype="int32"):
        if name in self.events:
            raise ValueError(f"Duplicate event: {name}")

        e = EventSpec(
            name=name,
            shape=shape,
            init=init,
            dtype=dtype,
        )
        self.events[name] = e
        return e

    def tile(
        self,
        name: str,
        impl: TileImpl,
        tile_num: tuple,
        *,
        attrs: dict[str, Any] | None = None,
        reads: list[TensorSpec] | None = None,
        writes: list[TensorSpec] | None = None,
        waits: list[DepSpec] | None = None,
        notifies: list[DepSpec] | None = None,
    ):
        if any(t.name == name for t in self.tiles):
            raise ValueError(f"Duplicate tile: {name}")

        tile = TileSpec(
            name=name,
            impl=impl,
            tile_num=tile_num,
            attrs=attrs or {},
            reads=reads or [],
            writes=writes or [],
            waits=waits or [],
            notifies=notifies or [],
        )

        self.tiles.append(tile)
        return tile

    def validate(self):
        errors: list[str] = []

        registered_tensors = tuple(self.tensors.values())
        registered_events = tuple(self.events.values())

        for tile in self.tiles:
            if not isinstance(tile.tile_num, tuple) or len(tile.tile_num) != 3:
                errors.append(
                    f"Tile {tile.name} tile_num must be a 3-element tuple, got {tile.tile_num!r}"
                )

            for tensor in tile.reads:
                if tensor not in registered_tensors:
                    errors.append(f"Tile {tile.name} reads unregistered tensor {tensor.name}")

            for tensor in tile.writes:
                if tensor not in registered_tensors:
                    errors.append(f"Tile {tile.name} writes unregistered tensor {tensor.name}")

            for dep in tile.waits:
                if dep.event not in registered_events:
                    errors.append(f"Tile {tile.name} waits on unregistered event {dep.event.name}")
                if not callable(dep.coord_map):
                    errors.append(
                        f"Tile {tile.name} wait coord_map for {dep.event.name} is not callable"
                    )
                if dep.expected is None:
                    errors.append(f"Tile {tile.name} wait on {dep.event.name} must specify expected")

            for dep in tile.notifies:
                if dep.event not in registered_events:
                    errors.append(f"Tile {tile.name} notifies unregistered event {dep.event.name}")
                if not callable(dep.coord_map):
                    errors.append(
                        f"Tile {tile.name} notify coord_map for {dep.event.name} is not callable"
                    )

        if errors:
            raise ValueError("Invalid megakernel spec:\n" + "\n".join(f"- {e}" for e in errors))

        return self

    def lower(self):
        self.validate()
        return {
            "name": self.name,
            "tensors": self.tensors,
            "events": self.events,
            "tiles": self.tiles,
        }
