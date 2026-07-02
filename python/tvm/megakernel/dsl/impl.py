from abc import ABC, abstractmethod


# ============================================================
# Tile implementation
# ============================================================


class TileImpl(ABC):
    """Local implementation of one tile kind.

    A `TileImpl` describes how one tile instance runs.  It should not describe
    global scheduling, event layout, or cross-tile dependency policy.  Those
    relationships are represented by `TileSpec`, `EventSpec`, and
    `DependencySpec`.
    """

    @classmethod
    def class_name(cls):
        """Return a stable implementation name."""

        return cls.__name__

    def __init__(self):
        self._instance_id = id(self)

    def __str__(self):
        return f"{self.__class__.__name__}-{self._instance_id:x}"

    # ===========================================================
    # User override hooks
    @classmethod
    def init_shared_resources(cls):
        """Initialize resources shared by all instances of this tile class."""

    @classmethod
    def finalize_shared_resources(cls):
        """Release resources created by `init_shared_resources`."""

    def device_init(self):
        """Initialize device-side state owned by one tile instance."""

    def host_init(self):
        """Initialize host-side state for one tile instance."""

    def prefetch(self, m_idx, n_idx, k_idx):
        """Optionally prefetch data for one tile instance before `run`."""

    @abstractmethod
    def run(self, m_idx, n_idx, k_idx):
        """Run one logical tile instance at index `(m_idx, n_idx, k_idx)`."""
