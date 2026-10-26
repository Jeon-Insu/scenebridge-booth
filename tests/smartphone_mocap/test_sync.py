from smartphone_mocap.sync import FrameSynchronizer
from smartphone_mocap.types import FramePacket


def packet(camera_id: str, timestamp: int, number: int = 0) -> FramePacket:
    return FramePacket(camera_id=camera_id, frame_number=number, capture_time_ns=timestamp)


def test_synchronizer_drops_stale_frame_then_pairs_all_cameras():
    synchronizer = FrameSynchronizer(["a", "b", "c"], tolerance_ns=10)
    synchronizer.push(packet("a", 50))
    synchronizer.push(packet("a", 100, 1))
    synchronizer.push(packet("b", 104))
    synchronizer.push(packet("c", 108))
    result = synchronizer.pop_synced()
    assert result is not None
    assert {name: item.capture_time_ns for name, item in result.items()} == {"a": 100, "b": 104, "c": 108}
    assert synchronizer.dropped_frames["a"] == 1


def test_synchronizer_applies_configured_offset():
    synchronizer = FrameSynchronizer(
        ["a", "b"], tolerance_ns=2, time_offsets_ns={"b": -100}
    )
    synchronizer.push(packet("a", 1000))
    synchronizer.push(packet("b", 1101))
    assert synchronizer.pop_synced() is not None


def test_synchronizer_can_continue_with_active_camera_subset():
    synchronizer = FrameSynchronizer(["a", "b", "c"], tolerance_ns=10)
    synchronizer.push(packet("a", 100))
    synchronizer.push(packet("b", 104))
    result = synchronizer.pop_synced(["a", "b"])
    assert result is not None
    assert set(result) == {"a", "b"}


def test_synchronizer_rejects_unknown_active_camera():
    synchronizer = FrameSynchronizer(["a", "b"], tolerance_ns=10)
    try:
        synchronizer.pop_synced(["a", "missing"])
    except KeyError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected unknown camera to raise KeyError")
