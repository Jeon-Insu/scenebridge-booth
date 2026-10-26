# tests/smartphone_mocap/test_booth_pipeline_parts.py
from smartphone_mocap.object_reconstruction import StaticObjectTracker
from smartphone_mocap.publisher import make_tracking_message
from smartphone_mocap.recognition.objects import keep_counted
from smartphone_mocap.types import SceneObject3D


def _obs(label, x, cams, conf=.9):
    return SceneObject3D("", label, (x, .3, 0.), conf, 1., cams)


def test_keep_counted_keeps_the_best_sightings_and_drops_unlisted_labels():
    kept = keep_counted([_obs("giraffe", .2, ("a", "b")), _obs("giraffe", .5, ("a", "b", "c")),
                         _obs("zebra", .8, ("a", "b")), _obs("rhino", .9, ("a", "b"))],
                        {"giraffe": 1, "zebra": 1})
    assert sorted((o.label, o.xyz[0]) for o in kept) == [("giraffe", .5), ("zebra", .8)]


def test_objects_vanish_by_time_only_when_hold_frames_is_none():
    tracker = StaticObjectTracker(min_confirmations=1, hold_frames=None, hold_seconds=1.5, reid_seconds=None)
    tracker.update([_obs("giraffe", .3, ("a", "b"))], timestamp_ns=0)
    for i in range(1, 60):                                   # 60 empty frames within 1.4 s
        assert tracker.update([], timestamp_ns=i * 23_000_000)
    assert tracker.update([], timestamp_ns=1_600_000_000) == []


def test_message_carries_yaw_pose_and_display_name_only_when_known():
    from dataclasses import replace
    turned = replace(_obs("giraffe", .3, ("a", "b")), global_id="g", yaw_deg=90.0, pose="upright",
                     display_name="기린")
    plain = replace(_obs("paper_cup", .6, ("a", "b")), global_id="c")
    objects = make_tracking_message(0, 0, [], [turned, plain], None)["objects"]
    assert (objects[0]["yaw_deg"], objects[0]["pose"], objects[0]["display_name"]) == (90.0, "upright", "기린")
    assert "yaw_deg" not in objects[1] and "display_name" not in objects[1]
