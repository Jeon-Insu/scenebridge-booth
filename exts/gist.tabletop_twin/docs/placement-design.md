# Placement decision — 2026-09-08

요약: LLM이 사용자의 의도에 맞춰 접촉 처리 옵션을 선택하고, 하네스가 USD 형상으로
보정량을 계산한다. 기본값은 위치 유지이며 공중 객체를 자동으로 테이블에 붙이지 않는다.
관통 가능성은 보정과 별개로 검사한다. 향후 넘어짐·굴러감·쌓기까지 판단해야 할 때는
물리 특성을 부여한 별도 시뮬레이션으로 후보 배치를 평가하는 방식을 검토한다.
**현재 물리 시뮬레이션은 구현하지 않았으며, 모든 편집을 물리로 처리하기로 확정한 것도 아니다.**

## Decision: intent from the LLM, geometry from the harness

The demo does **not** automatically snap rotated or scaled objects to the table.
The LLM selects a placement policy from the user's intent and the current USD
contact facts. Code computes the requested transform and any selected correction.
This is virtual scene editing, not motion of the physical camera-observed objects.

| Policy | Result |
| --- | --- |
| `preserve_position` (default) | Apply rotation/scale around the existing asset origin without changing its position. Report possible penetration; do not fix it silently. |
| `maintain_support` | Only if the current geometry is already in table contact, adjust Z after the requested transform to retain contact. Refuse an airborne/unknown state. |
| `place_on_table` | Explicitly land the transformed geometry on the table, retaining XY, orientation and size. May lower or raise the asset origin. |

`rotate_object` and `scale_object` accept `placement`. `transform_object` combines
rotation and uniform scale in one operation with one solve. `place_on_table` is
also a separate action. `inspect_table_contact` and `list_objects` expose the
current clearance, footprint, possible penetration and previous chosen policy.

The previous policy is **metadata, not a persistent support constraint**. A later
lift followed by rotation defaults to preserving the airborne pivot. A command
does not enable per-frame grounding. If intent is ambiguous, preserve position
or offer a preview. Never infer metric geometry from an LLM's image description.

Examples:

- “공중에 있는 컵을 X축으로 90도 회전해 줘”: `preserve_position`.
- “테이블에 닿은 상태를 유지하면서 컵을 X축으로 90도 돌려 줘”: `maintain_support`.
- “컵을 두 배로 키우고 테이블 위에 눕혀 줘”: combined `transform_object`, `place_on_table`.
- “컵을 테이블 위에 내려놔”: `place_on_table`; **not** `change_height(height_m=0)`.
- `change_height` changes asset-origin Z. Origin Z and a rotated mesh's lower
  surface clearance are not the same. Lifting does not preserve a support lock.

## Geometric implementation and limits

`smartphone_mocap/table_contact.py` is a pure deterministic backend shared with
Kit. `gist.tabletop_twin/contact.py` reads actual Asset/Proxy boundable descendants
through `UsdGeom.BBoxCache.ComputeRelativeBound`. It exports each leaf's bound
corners in the managed object's frame, excluding preview markers and guides.
The solver applies scale and quaternion rotation, computes the resulting lower
bound, and, **only for an explicitly selected contact policy**, solves
`origin_z += -minimum_geometry_z` (TableFrame surface Z=0).

The agent predicts the result from a snapshot; the extension independently
revalidates against current USD geometry before mutating any object. Geometry or
policy failure rejects the action. Preview and actual application share the
same solver, and System messages report final XYZ, clearance and warnings.
All TRS actions are checked even when no correction was selected. Missing
geometry permits uncorrected transforms with an explicit unknown result, but
cannot support a contact solve.

Current assumptions and limitations:

- Fixed, horizontal 1.25 × 0.75 m tabletop, TableFrame Z=0; world offset and rig
  yaw do not change the local vertical contact calculation.
- Contact classification tolerance: 5 mm (geometric tolerance, not camera accuracy).
- Correction requires the entire conservative XY footprint inside the table.
- Bounds are conservative. A tilted cylinder, concave object or sparse mesh can
  have an empty bounding-box corner below its actual surface: the result may
  float slightly. “Possible penetration” is not an exact mesh collision result.
- No stacking on other objects, center-of-mass stability, friction, rolling,
  falling, or physical collision response. This is not a physics simulation.
- Existing live edits remain raw tracked position + edit offset. Repeated equal
  packets do not accumulate correction, but raw tracking jitter still passes
  through; this change does not replace tracking with a fixed virtual pose.
- Contact choices are not automatically repeated when tracking updates or when
  a reset restores the original transform. A subsequent contact action is explicit.
- Runtime state, preview IDs and virtual offsets have their existing session/track
  lifetimes; a deleted/recreated track does not inherit the former track's edit.
- Rule-based fallback does not replace the LLM's contextual policy selection.
  Its unqualified rotation/scale default remains position preservation.

USD documentation: [BBoxCache and cache invalidation](https://openusd.org/release/api/class_usd_geom_b_box_cache.html).
The adapter creates fresh caches for each inspection, avoiding stale transform bounds.

## Future option: physics-backed evaluation (not implemented)

Physics may be the better backend when the question becomes “will this stand,
fall, roll, collide or stack?” It is not automatically preferable for every edit:
an intentional airborne pose must not be released under gravity without a request.
The LLM should continue choosing intent, while a physics backend evaluates it.

Proposed extension path:

1. Configure table/object collision shapes, rigid-body types, mass, center of
   mass, friction and restitution. Choose suitable approximations for generated
   meshes; visual geometry alone is insufficient.
2. Add an explicit operation such as `simulate_placement` / `release_object`,
   keeping “hold in air” distinct from “release under gravity”.
3. Evaluate a copy of the candidate scene in an isolated simulation. Use bounded
   simulated time, a settling threshold and a timeout; report stable/unstable/
   unresolved along with the final pose and any collisions.
4. Show the result as a preview and commit the selected outcome. Do not advance
   or alter the audience's live scene merely to answer a question.
5. Decide pose ownership: perception, manual editing or physics must be the
   single writer for each object's transform at a given time. Tracking packets
   must not overwrite a dynamic rigid body's result on every update.
6. Compare accuracy, latency and repeatability against the present geometric
   solver before making simulation the default for any class of command.

Keep the external intent/placement API stable where useful; substitute or add a
physics evaluator behind it. Physics requires authored assumptions and validation,
not just a `simulate=True` flag. See NVIDIA's [rigid-body setup documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html).

## Verification

- Pure math/agent tests: default no-snap + warnings, support-preserving tilt and
  double scale, shrink/upright recovery, airborne pivot preservation, landing,
  invalid/missing geometry, edge refusal, offset asset origins and preview/apply.
- Isaac headless test: actual referenced mug bounds, world/table frame agreement,
  preview and commit, fresh snapshot parity, live packets without accumulated
  corrections, refused airborne support, and penetration warnings.
- Manual demo: request the four example commands above and verify System messages
  plus the viewport. Contact warnings and replies are advisory, not proof of
  physical stability.
