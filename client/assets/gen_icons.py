#!/usr/bin/env python3
"""Generate tray icon RGBA files (4 bytes/pixel: R G B A).

Emits two sizes:
  - 24x24 (*.rgba)      used by the X11 XEMBED tray
  - 48x48 (*_lg.rgba)   used by the Wayland StatusNotifierItem tray,
                        where compositors typically render larger than 24px
"""
import math, struct

def aa_fill(dist_from_edge):
    return max(0.0, min(1.0, dist_from_edge + 0.5))

def aa_ring(dist, radius, half_w):
    outer = aa_fill(radius + half_w - dist)
    inner = aa_fill(dist - (radius - half_w))
    return outer * inner

def gen_icon(filename, shape_fn, r, g, b, size):
    data = bytearray()
    cx = cy = size / 2.0
    scale = size / 24.0
    for y in range(size):
        for x in range(size):
            px, py = x + 0.5, y + 0.5
            dx, dy = px - cx, py - cy
            dist = math.sqrt(dx * dx + dy * dy)
            a = shape_fn(dist, size, scale)
            data += struct.pack("BBBB", int(r * a), int(g * a), int(b * a), int(a * 255))
    with open(filename, "wb") as f:
        f.write(data)
    print(f"  {filename}: {len(data)} bytes")

def filled(dist, s, k):
    return aa_fill(s / 2.0 - 3.0 * k - dist)

def filled_lg(dist, s, k):
    return aa_fill(s / 2.0 - 2.0 * k - dist)

def ring(dist, s, k):
    return aa_ring(dist, s / 2.0 - 4.0 * k, 1.8 * k)

def bullseye(dist, s, k):
    outer = aa_ring(dist, s / 2.0 - 3.5 * k, 1.5 * k)
    inner = aa_fill(2.5 * k - dist)
    return 1.0 - (1.0 - outer) * (1.0 - inner)

ICONS = [
    ("idle",                 ring,      0x88, 0x88, 0x88),
    ("sleeping",             filled,    0x88, 0x88, 0x88),
    ("listening",            ring,      0x4C, 0xAF, 0x50),
    ("dictating",            filled,    0x4C, 0xAF, 0x50),
    ("recording",            filled_lg, 0xF4, 0x43, 0x36),
    ("processing",           bullseye,  0x4C, 0xAF, 0x50),
    ("recording_processing", bullseye,  0xF4, 0x43, 0x36),
]

if __name__ == "__main__":
    for size, suffix in [(24, ""), (48, "_lg")]:
        print(f"Generating {size}x{size} tray icons...")
        for name, shape, r, g, b in ICONS:
            gen_icon(f"{name}{suffix}.rgba", shape, r, g, b, size)
    print("Done.")
