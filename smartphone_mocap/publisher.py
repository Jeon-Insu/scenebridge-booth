from __future__ import annotations

import json
import socket
from pathlib import Path

from .types import Person3D, SceneObject3D


def make_tracking_message(
    timestamp_ns: int,
    frame_number: int,
    people: list[Person3D],
    objects: list[SceneObject3D] | None = None,
    rig_to_scene: dict | None = None,
) -> dict:
    encoded_people = []
    for person in people:
        root = person.root_xyz()
        encoded_people.append(
            {
                "id": person.global_id,
                "state": person.state,
                "root_xyz": list(root) if root else None,
                "joints": {
                    name: {
                        "xyz": [joint.x, joint.y, joint.z],
                        "confidence": joint.confidence,
                        "reprojection_error_px": joint.reprojection_error_px,
                        "source_cameras": list(joint.source_cameras),
                    }
                    for name, joint in person.joints.items()
                },
            }
        )
    encoded_objects = [
        {
            "id": item.global_id,
            "label": item.label,
            "state": item.state,
            "xyz": list(item.xyz),
            "confidence": item.confidence,
            "reprojection_error_px": item.reprojection_error_px,
            "source_cameras": list(item.source_cameras),
            "source_tracks": [
                {"camera_id": camera_id, "local_track_id": local_track_id}
                for camera_id, local_track_id in item.source_track_keys
            ],
            "asset_uri": item.asset_uri,
            "asset_key": item.asset_key,
            "position_mode": item.position_mode,
            **({"yaw_deg": round(float(item.yaw_deg), 1)} if item.yaw_deg is not None else {}),
            **({"pose": item.pose} if item.pose else {}),
            **({"display_name": item.display_name} if item.display_name else {}),
        }
        for item in (objects or [])
    ]
    message = {
        "schema": "gist.smartphone_mocap.v1",
        "timestamp_ns": int(timestamp_ns),
        "frame_number": int(frame_number),
        "coordinate_frame": "BoothFrame",
        "people": encoded_people,
        "objects": encoded_objects,
    }
    if rig_to_scene:
        message["rig_to_scene"] = rig_to_scene
    return message


class UdpJsonPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 7001) -> None:
        self.address = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, message: dict) -> None:
        payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.socket.sendto(payload, self.address)

    def close(self) -> None:
        self.socket.close()


class JsonLinesWriter:
    def __init__(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.file = destination.open("w", encoding="utf-8")

    def publish(self, message: dict) -> None:
        self.file.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()
