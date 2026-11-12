"""Generate identity mark assets for Credio:
- favicon.svg
- favicon.ico (multi-size 16x16, 32x32)
- apple-touch-icon.png (180x180)
- favicon.png (32x32)

Constraint: All <= 10KB total.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw

STATIC_DIR = Path(r"D:\Clones\Transit\credio\app\static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

SVG_CONTENT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32" fill="none">
  <rect width="32" height="32" rx="6" fill="#0f172a"/>
  <!-- Decision Gauge Arc -->
  <path d="M 8.2 22.5 A 9 9 0 1 1 23.8 22.5" stroke="#f8fafc" stroke-width="1.8" stroke-linecap="round"/>
  <!-- Calibration Benchmark Scale Ticks -->
  <path d="M 16 9 V 11" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>
  <path d="M 10.5 14.5 L 12 15.5" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>
  <path d="M 21.5 14.5 L 20 15.5" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>
  <!-- Calibrated Decision Threshold Needle -->
  <path d="M 16 18 L 20.5 10.5" stroke="#38bdf8" stroke-width="2.0" stroke-linecap="round"/>
  <!-- Decision Pivot Node -->
  <circle cx="16" cy="18" r="2" fill="#38bdf8"/>
  <!-- Horizontal Baseline -->
  <path d="M 7 26 H 25" stroke="#64748b" stroke-width="1.2" stroke-linecap="round"/>
</svg>
"""


def draw_credio_mark(size: int = 512, rounded_radius: int = 96) -> Image.Image:
    """Draw high-resolution Credio decision gauge mark with supersampling."""
    scale = 4
    w = size * scale
    r = rounded_radius * scale

    img = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rectangle (Slate 900 #0f172a)
    draw.rounded_rectangle([0, 0, w, w], radius=r, fill=(15, 23, 42, 255))

    s = w / 32.0
    cx, cy = 16.0 * s, 18.0 * s
    radius = 9.0 * s

    # 1. Gauge Arc: 210 deg to 330 deg (in standard screen coords: 210 to -30 / 330)
    # Using small polyline segments for crisp geometric rendering
    arc_pts = []
    # angles from 150 deg down through 90, 0, to -30 (counter-clockwise in math, clockwise in screen)
    # Start at bottom-left: 210 deg in math -> screen angle from x-axis: 150 deg
    for deg in range(-30, 211, 2):
        rad = math.radians(deg)
        # Note: In screen coords, y increases downwards.
        # Top of dial is at screen angle 90 (or math 90: y = cy - radius)
        px = cx + radius * math.cos(rad)
        py = cy - radius * math.sin(rad)
        arc_pts.append((px, py))

    # Arc spans from 210 deg (bottom left) to -30 deg (bottom right)
    gauge_pts = []
    for deg in range(-30, 211, 3):
        rad = math.radians(deg)
        gauge_pts.append((cx - radius * math.cos(rad), cy - radius * math.sin(rad)))

    # Draw arc with width 1.8 * s
    draw.line(gauge_pts, fill=(248, 250, 252, 255), width=int(1.8 * s), joint="round")

    # 2. Calibration Ticks
    # Top tick: (16, 9) to (16, 11)
    draw.line([(16.0 * s, 9.0 * s), (16.0 * s, 11.0 * s)], fill=(148, 163, 184, 255), width=int(1.2 * s))
    # Left tick: (10.5, 14.5) to (12.0, 15.5)
    draw.line([(10.5 * s, 14.5 * s), (12.0 * s, 15.5 * s)], fill=(148, 163, 184, 255), width=int(1.2 * s))
    # Right tick: (21.5, 14.5) to (20.0, 15.5)
    draw.line([(21.5 * s, 14.5 * s), (20.0 * s, 15.5 * s)], fill=(148, 163, 184, 255), width=int(1.2 * s))

    # 3. Baseline: (7, 26) to (25, 26)
    draw.line([(7.0 * s, 26.0 * s), (25.0 * s, 26.0 * s)], fill=(100, 116, 139, 255), width=int(1.2 * s))

    # 4. Calibrated Decision Needle: (16, 18) to (20.5, 10.5)
    draw.line([(16.0 * s, 18.0 * s), (20.5 * s, 10.5 * s)], fill=(56, 189, 248, 255), width=int(2.0 * s), joint="round")

    # 5. Center Pivot Node
    cr = 2.0 * s
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(56, 189, 248, 255))

    return img.resize((size, size), Image.Resampling.LANCZOS)


def main():
    # 1. Write SVG
    svg_path = STATIC_DIR / "favicon.svg"
    svg_path.write_text(SVG_CONTENT.strip(), encoding="utf-8")
    svg_sz = svg_path.stat().st_size
    print(f"favicon.svg: {svg_sz} bytes")

    # 2. Apple Touch Icon (180x180) - optimize PNG
    apple_img = draw_credio_mark(180, rounded_radius=32)
    # Convert to adaptive 64-color palette for tiny byte footprint
    apple_opt = apple_img.convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
    apple_path = STATIC_DIR / "apple-touch-icon.png"
    apple_opt.save(apple_path, format="PNG", optimize=True)
    apple_sz = apple_path.stat().st_size
    print(f"apple-touch-icon.png: {apple_sz} bytes ({apple_sz/1024:.2f} KB)")

    # 3. Favicon.png (32x32) for direct Streamlit page_icon support
    fav_32 = draw_credio_mark(32, rounded_radius=6)
    fav_32_opt = fav_32.convert("P", palette=Image.Palette.ADAPTIVE, colors=32)
    fav_png_path = STATIC_DIR / "favicon.png"
    fav_32_opt.save(fav_png_path, format="PNG", optimize=True)
    fav_png_sz = fav_png_path.stat().st_size
    print(f"favicon.png (32x32): {fav_png_sz} bytes ({fav_png_sz/1024:.2f} KB)")

    # 4. Multi-size favicon.ico (16x16, 32x32)
    ico_img = draw_credio_mark(64, rounded_radius=12)
    ico_path = STATIC_DIR / "favicon.ico"
    ico_img.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32)])
    ico_sz = ico_path.stat().st_size
    print(f"favicon.ico: {ico_sz} bytes ({ico_sz/1024:.2f} KB)")

    total_key_assets = svg_sz + apple_sz + ico_sz
    print(f"\n--- Total Identity Mark Package (SVG + Apple + ICO): {total_key_assets} bytes ({total_key_assets/1024:.2f} KB) ---")
    assert total_key_assets <= 10 * 1024, f"Package exceeds 10KB budget: {total_key_assets} bytes"
    print("GATE 1 PASS: All mark assets <= 10KB total!")


if __name__ == "__main__":
    main()
