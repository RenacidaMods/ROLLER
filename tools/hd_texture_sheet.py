"""Stitch dumped texture tiles into one sheet for upscaling, then cut it back.

The "Dump Textures as PNG" debug option writes each 64x64 tile of a texture
bank to ``hd_textures_dump/<bank>/<tile>.png``. Many track and sign images
span several neighbouring tiles, so upscaling tiles one at a time leaves seams
where the halves meet. This tool joins a whole bank into one sheet, which can
be upscaled in any tool, and cuts the result back into HD tiles.

    python tools/hd_texture_sheet.py stitch D:/GAMES/ROLLER/hd_textures_dump/tracksix
        -> D:/GAMES/ROLLER/hd_textures_work/tracksix.png (+ tracksix.json layout)

    (upscale tracksix.png by 2x, 4x, ... in any tool, keep the file anywhere)

    python tools/hd_texture_sheet.py cut D:/GAMES/ROLLER/hd_textures_work/tracksix.json UPSCALED.png
        -> D:/GAMES/ROLLER/hd_textures/tracksix/000.png ...

Sheet layout: tiles in row-major order, ``cols`` per row, with ``gap`` pixels
between rows filled by repeating each row's edge pixels, so an upscaler never
blends a tile with the unrelated tile underneath it. See-through pixels
(palette index 0, alpha 0) get their colour filled in from nearby opaque
pixels, so upscalers don't drag the hidden colour into visible edges.

On cutting, alpha is snapped back to fully on or off at 50%, matching how the
game treats transparency. If the upscaled sheet has no alpha channel, the
original tile's alpha is scaled up instead.

Needs Pillow only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

TILE = 64
LAYOUT_VERSION = 1


def tile_files(bank_dir: Path) -> dict[int, Path]:
    tiles = {}
    for path in bank_dir.glob("*.png"):
        if path.stem.isdigit():
            tiles[int(path.stem)] = path
    return tiles


def bleed_transparent(tile: Image.Image, passes: int = 4) -> Image.Image:
    """Fill see-through pixels' colour from opaque neighbours, a few pixels deep."""
    rgba = tile.convert("RGBA")
    raw = rgba.tobytes()
    px = [tuple(raw[i:i + 4]) for i in range(0, len(raw), 4)]
    w, h = rgba.size
    known = [a >= 128 for (_, _, _, a) in px]
    colours = [(r, g, b) for (r, g, b, _) in px]
    for _ in range(passes):
        newly = []
        for i, ok in enumerate(known):
            if ok:
                continue
            x, y = i % w, i // w
            total = [0, 0, 0]
            count = 0
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < w and 0 <= ny < h and known[ny * w + nx]:
                    c = colours[ny * w + nx]
                    total[0] += c[0]
                    total[1] += c[1]
                    total[2] += c[2]
                    count += 1
            if count:
                newly.append((i, (total[0] // count, total[1] // count, total[2] // count)))
        if not newly:
            break
        for i, c in newly:
            colours[i] = c
            known[i] = True
    return Image.frombytes("RGBA", (w, h),
                           bytes(v for c, p in zip(colours, px) for v in (c[0], c[1], c[2], p[3])))


def stitch(bank_dir: Path, out_dir: Path | None, cols: int, gap: int) -> int:
    tiles = tile_files(bank_dir)
    if not tiles:
        print(f"No numbered PNG tiles found in {bank_dir}", file=sys.stderr)
        return 1
    count = max(tiles) + 1
    missing = [i for i in range(count) if i not in tiles]
    rows = (count + cols - 1) // cols
    row_pitch = TILE + gap
    sheet = Image.new("RGBA", (cols * TILE, rows * TILE + (rows - 1) * gap), (0, 0, 0, 0))

    for index in range(count):
        if index not in tiles:
            continue
        tile = Image.open(tiles[index])
        if tile.size != (TILE, TILE):
            print(f"{tiles[index]} is {tile.size[0]}x{tile.size[1]}, expected {TILE}x{TILE}", file=sys.stderr)
            return 1
        sheet.paste(bleed_transparent(tile), ((index % cols) * TILE, (index // cols) * row_pitch))

    # Fill each gap by repeating the edge rows of the tiles above and below it.
    top_half = gap // 2
    for row in range(rows - 1):
        gap_y = row * row_pitch + TILE
        above = sheet.crop((0, gap_y - 1, sheet.width, gap_y))
        below = sheet.crop((0, gap_y + gap, sheet.width, gap_y + gap + 1))
        for y in range(gap):
            sheet.paste(above if y < top_half else below, (0, gap_y + y))

    if out_dir is None:
        out_dir = bank_dir.resolve().parent.parent / "hd_textures_work"
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = bank_dir.resolve().name
    sheet_path = out_dir / f"{bank}.png"
    layout_path = out_dir / f"{bank}.json"
    sheet.save(sheet_path)
    layout_path.write_text(json.dumps({
        "version": LAYOUT_VERSION,
        "bank": bank,
        "source": str(bank_dir.resolve()),
        "tiles": count,
        "missing": missing,
        "tile_size": TILE,
        "cols": cols,
        "gap": gap,
        "width": sheet.width,
        "height": sheet.height,
    }, indent=2) + "\n")

    print(f"Wrote {sheet_path} ({sheet.width}x{sheet.height}, {count} tiles, {cols} per row)")
    print(f"Wrote {layout_path}")
    if missing:
        print(f"Note: tiles {missing} were missing and left blank")
    return 0


def parse_tiles(spec: str | None, count: int) -> list[int]:
    if not spec:
        return list(range(count))
    chosen = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            first, last = part.split("-", 1)
            chosen.update(range(int(first), int(last) + 1))
        elif part:
            chosen.add(int(part))
    return sorted(i for i in chosen if 0 <= i < count)


def cut(layout_path: Path, upscaled_path: Path, out_root: Path | None,
        tile_spec: str | None, alpha_mode: str) -> int:
    layout = json.loads(layout_path.read_text())
    if layout.get("version") != LAYOUT_VERSION:
        print(f"{layout_path} has an unknown layout version", file=sys.stderr)
        return 1

    sheet = Image.open(upscaled_path)
    scale_x = sheet.width / layout["width"]
    scale_y = sheet.height / layout["height"]
    scale = round(scale_x)
    if scale < 1 or abs(scale_x - scale) > 1e-6 or abs(scale_y - scale) > 1e-6:
        print(f"{upscaled_path} is {sheet.width}x{sheet.height}, which isn't a whole-number "
              f"scale of the original {layout['width']}x{layout['height']} sheet", file=sys.stderr)
        return 1
    if scale & (scale - 1):
        print(f"Warning: {scale}x isn't a power of two; 2x, 4x or 8x is safest for the game")

    has_alpha = "A" in sheet.getbands()
    if alpha_mode == "auto":
        alpha_mode = "upscaled" if has_alpha else "original"
    if alpha_mode == "upscaled" and not has_alpha:
        print(f"{upscaled_path} has no alpha channel; use --alpha original", file=sys.stderr)
        return 1
    sheet = sheet.convert("RGBA")

    bank = layout["bank"]
    source = Path(layout["source"])
    if out_root is None:
        out_root = source.parent.parent / "hd_textures"
    out_dir = out_root / bank
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_px = layout["tile_size"] * scale
    row_pitch = (layout["tile_size"] + layout["gap"]) * scale
    written = 0
    for index in parse_tiles(tile_spec, layout["tiles"]):
        if index in layout["missing"]:
            continue
        x = (index % layout["cols"]) * tile_px
        y = (index // layout["cols"]) * row_pitch
        tile = sheet.crop((x, y, x + tile_px, y + tile_px))
        r, g, b, a = tile.split()
        if alpha_mode == "original":
            original = Image.open(source / f"{index:03d}.png").convert("RGBA")
            a = original.getchannel("A").resize((tile_px, tile_px), Image.NEAREST)
        a = a.point(lambda v: 255 if v >= 128 else 0)
        Image.merge("RGBA", (r, g, b, a)).save(out_dir / f"{index:03d}.png")
        written += 1

    print(f"Wrote {written} tiles at {tile_px}x{tile_px} ({scale}x) to {out_dir} "
          f"(alpha from {alpha_mode} image)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_stitch = sub.add_parser("stitch", help="join a dumped bank folder into one sheet")
    p_stitch.add_argument("bank_dir", type=Path, help="e.g. hd_textures_dump/tracksix")
    p_stitch.add_argument("--out", type=Path, default=None,
                          help="output folder (default: hd_textures_work next to hd_textures_dump)")
    p_stitch.add_argument("--cols", type=int, default=16, help="tiles per row (default 16)")
    p_stitch.add_argument("--gap", type=int, default=8, help="padding pixels between rows (default 8)")

    p_cut = sub.add_parser("cut", help="cut an upscaled sheet back into HD tiles")
    p_cut.add_argument("layout", type=Path, help="the .json written by stitch")
    p_cut.add_argument("upscaled", type=Path, help="the upscaled sheet image")
    p_cut.add_argument("--out", type=Path, default=None,
                       help="output root (default: hd_textures next to hd_textures_dump)")
    p_cut.add_argument("--tiles", default=None, help="only these tiles, e.g. 12,40-43 (default: all)")
    p_cut.add_argument("--alpha", choices=("auto", "upscaled", "original"), default="auto",
                       help="where transparency comes from (default: upscaled if it has alpha)")

    args = parser.parse_args()
    if args.command == "stitch":
        if args.cols < 1 or args.gap < 0:
            parser.error("--cols must be at least 1 and --gap can't be negative")
        return stitch(args.bank_dir, args.out, args.cols, args.gap)
    return cut(args.layout, args.upscaled, args.out, args.tiles, args.alpha)


if __name__ == "__main__":
    sys.exit(main())
