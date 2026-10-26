#!/usr/bin/env bash
# Bandwidth correction for the two cameras that share one USB2 hub (found automatically).
# Changes only their video-driver bindings. No hub/audio/keyboard reset,
# module unload, persistent modprobe configuration or camera pose changes.
# Linux v6.17: UVC_QUIRK_FIX_BANDWIDTH (128) applies to uncompressed video.
# Device quirks are copied at probe, so restoring the global parameter alone
# does NOT restore an already-probed camera: use this script's restore action.
set -euo pipefail
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH

action=${1:-check}
if (( $# > 1 )) || [[ ! "$action" =~ ^(check|apply|restore)$ ]]; then
    printf 'Usage: bash tools/creative_uvc_bandwidth.sh {check|apply|restore}\n' >&2
    exit 2
fi

parameter=/sys/module/uvcvideo/parameters/quirks
driver=/sys/bus/usb/drivers/uvcvideo
# The pair is found, not fixed: the two Creative cameras that share one USB bus (one hub).
# Re-cabling therefore needs no edit here; any other arrangement is refused.
declare -A per_bus=()
for base in /sys/bus/usb/devices/*; do
    name=${base##*/}
    [[ "$name" =~ ^[0-9]+-[0-9.]+$ && -r "$base/idVendor" && -r "$base/idProduct" ]] || continue
    [[ "$(<"$base/idVendor")" == 041e && "$(<"$base/idProduct")" == 40a0 ]] || continue
    per_bus[${name%%-*}]+="$name "
done
devices=()
for bus in "${!per_bus[@]}"; do
    read -r -a found <<< "${per_bus[$bus]}"
    if (( ${#found[@]} > 2 )) || { (( ${#found[@]} == 2 )) && (( ${#devices[@]} )); }; then
        printf 'Refusing: expected exactly one hub with two cameras. Cameras per bus:\n' >&2
        for b in "${!per_bus[@]}"; do printf '  bus %s: %s\n' "$b" "${per_bus[$b]}" >&2; done
        exit 1
    fi
    if (( ${#found[@]} == 2 )); then devices=("${found[@]}"); fi
done
if (( ${#devices[@]} != 2 )); then
    printf 'No hub shares two cameras; no correction is needed.\n'
    exit 0
fi
interfaces=("${devices[0]}:1.0" "${devices[1]}:1.0")

for device in "${devices[@]}"; do
    base=/sys/bus/usb/devices/$device
    if [[ ! -r "$base/idVendor" || ! -r "$base/idProduct" ]] ||
       [[ "$(<"$base/idVendor")" != 041e || "$(<"$base/idProduct")" != 40a0 ]]; then
        printf 'Refusing: %s is not the expected Creative 041e:40a0 camera. Recheck mapping.\n' "$device" >&2
        exit 1
    fi
    interface=$base:1.0
    if [[ ! -r "$interface/bInterfaceClass" || ! -r "$interface/bInterfaceSubClass" ]] ||
       [[ "$(<"$interface/bInterfaceClass")" != 0e || "$(<"$interface/bInterfaceSubClass")" != 01 ]]; then
        printf 'Refusing: unexpected video-control interface %s.\n' "$interface" >&2
        exit 1
    fi
    if [[ -L "$interface/driver" ]] &&
       [[ "$(readlink -f "$interface/driver")" != "$driver" ]]; then
        printf 'Refusing: interface is owned by a different driver: %s\n' "$interface" >&2
        exit 1
    fi
    if [[ "$action" != restore && ! -L "$interface/driver" ]]; then
        printf 'Refusing: %s is unbound. Use restore first.\n' "$interface" >&2
        exit 1
    fi
    printf 'Verified %s: Creative video interface %s:1.0\n' "$device" "$device"
done

original=$(<"$parameter")
printf 'Global probe default: %s (not a per-camera effective-quirks readout).\n' "$original"
if [[ "$original" != 4294967295 ]]; then
    printf 'Refusing: global quirks differ from the verified automatic default.\n' >&2
    exit 1
fi
if [[ "$action" == check ]]; then
    printf 'Read-only check passed. No settings changed.\n'
    exit 0
fi
if (( EUID != 0 )); then
    printf 'Requires sudo in your own terminal. Do not share the password in chat.\n' >&2
    exit 1
fi

changed=()
completed=0
rebind_one() {
    local target=$1
    if [[ -L /sys/bus/usb/devices/$target/driver ]]; then
        printf '%s' "$target" > "$driver/unbind" || return 1
    fi
    printf '%s' "$target" > "$driver/bind"
}
cleanup() {
    local status=$? target
    trap - EXIT INT TERM HUP
    if ! printf '%s' "$original" > "$parameter"; then
        printf 'ERROR: could not restore global parameter; a reboot restores this temporary experiment.\n' >&2
        exit 1
    fi
    if (( ! completed )); then
        printf 'Interrupted/failed: attempting to restore default bindings.\n' >&2
        for target in "${changed[@]}"; do
            if ! rebind_one "$target"; then
                printf 'ERROR: rebind failed for %s; reconnect that camera or reboot.\n' "$target" >&2
                status=1
            fi
        done
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

value=$original
[[ "$action" != apply ]] || value=128
printf 'Rebinding the shared pair (%s %s) video only with probe quirks=%s; their streams will briefly disconnect.\n' "${devices[@]}" "$value"
printf '%s' "$value" > "$parameter"
for target in "${interfaces[@]}"; do
    changed+=("$target")
    rebind_one "$target"
done
printf '%s' "$original" > "$parameter"
completed=1
if [[ "$action" == apply ]]; then
    printf 'Shared pair rebound with bandwidth correction. Global default restored; other cameras untouched.\n'
    printf 'Next: test YUYV reception. MJPG is not affected by this correction.\n'
    printf 'Undo: sudo bash tools/creative_uvc_bandwidth.sh restore\n'
else
    printf 'Shared pair rebound with original automatic quirks.\n'
fi
