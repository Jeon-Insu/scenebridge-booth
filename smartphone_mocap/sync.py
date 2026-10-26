from __future__ import annotations

from collections import deque

from .types import FramePacket


class FrameSynchronizer:
    """Pairs the nearest timestamped frame from every configured camera."""

    def __init__(
        self,
        camera_ids: list[str],
        tolerance_ns: int,
        *,
        time_offsets_ns: dict[str, int] | None = None,
        max_queue_size: int = 30,
    ) -> None:
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("camera_ids must be unique")
        self.camera_ids = tuple(camera_ids)
        self.tolerance_ns = int(tolerance_ns)
        self.time_offsets_ns = dict(time_offsets_ns or {})
        self.max_queue_size = int(max_queue_size)
        self.queues = {camera_id: deque() for camera_id in self.camera_ids}
        self.dropped_frames = {camera_id: 0 for camera_id in self.camera_ids}

    def adjusted_time_ns(self, packet: FramePacket) -> int:
        return packet.capture_time_ns + self.time_offsets_ns.get(packet.camera_id, 0)

    def push(self, packet: FramePacket) -> None:
        if packet.camera_id not in self.queues:
            raise KeyError(f"Unknown camera: {packet.camera_id}")
        queue = self.queues[packet.camera_id]
        queue.append(packet)
        while len(queue) > self.max_queue_size:
            queue.popleft()
            self.dropped_frames[packet.camera_id] += 1

    def pop_synced(
        self,
        camera_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, FramePacket] | None:
        """Pop a synchronized set from all cameras or a currently active subset."""
        selected = tuple(camera_ids) if camera_ids is not None else self.camera_ids
        if not selected:
            return None
        unknown = set(selected) - set(self.camera_ids)
        if unknown:
            raise KeyError(f"Unknown cameras: {sorted(unknown)}")
        while all(self.queues[camera_id] for camera_id in selected):
            heads = {camera_id: self.queues[camera_id][0] for camera_id in selected}
            times = {camera_id: self.adjusted_time_ns(packet) for camera_id, packet in heads.items()}
            minimum_id = min(times, key=times.get)
            spread = max(times.values()) - min(times.values())
            if spread <= self.tolerance_ns:
                return {camera_id: self.queues[camera_id].popleft() for camera_id in selected}
            self.queues[minimum_id].popleft()
            self.dropped_frames[minimum_id] += 1
        return None
