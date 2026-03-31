#!/usr/bin/env python3
"""Generate tray icon RGBA files (24x24, 4 bytes/pixel: R G B A)."""
import math, struct

SIZE = 24

def aa_fill(dist_from_edge):
    return max(0.0, min(1.0, dist_from_edge + 0.5))

def aa_ring(dist, radius, half_w):
    outer = aa_fill(radius + half_w - dist)
    inner = aa_fill(dist - (radius - half_w))
    return outer * inner

def gen_icon(filename, shape_fn, r, g, b):
    data = bytearray()
    s = SIZE
    cx = cy = s / 2.0
    for y in range(s):
        for x in range(s):
            px, py = x + 0.5, y + 0.5
            dx, dy = px - cx, py - cy
            dist = math.sqrt(dx * dx + dy * dy)
            a = shape_fn(dist, s)
            data += struct.pack("BBBB", int(r * a), int(g * a), int(b * a), int(a * 255))
    with open(filename, "wb") as f:
        f.write(data)
    print(f"  {filename}: {len(data)} bytes")

def filled(dist, s):
    return aa_fill(s / 2.0 - 3.0 - dist)

def filled_lg(dist, s):
    return aa_fill(s / 2.0 - 2.0 - dist)

def ring(dist, s):
    return aa_ring(dist, s / 2.0 - 4.0, 1.8)

def bullseye(dist, s):
    outer = aa_ring(dist, s / 2.0 - 3.5, 1.5)
    inner = aa_fill(2.5 - dist)
    return 1.0 - (1.0 - outer) * (1.0 - inner)

if __name__ == "__main__":
    print(f"Generating {SIZE}x{SIZE} tray icons...")
    gen_icon("idle.rgba",                 ring,     0x88, 0x88, 0x88)
    gen_icon("listening.rgba",            ring,     0x4C, 0xAF, 0x50)
    gen_icon("dictating.rgba",            filled,   0x4C, 0xAF, 0x50)
    gen_icon("recording.rgba",            filled_lg, 0xF4, 0x43, 0x36)
    gen_icon("processing.rgba",           bullseye, 0x4C, 0xAF, 0x50)
    gen_icon("recording_processing.rgba", bullseye, 0xF4, 0x43, 0x36)
    print("Done.")
