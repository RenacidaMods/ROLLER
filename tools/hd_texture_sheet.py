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

With Upscayl installed (https://upscayl.org), two more commands drive its
command-line engine (upscayl-bin) directly, so no manual upscaling is needed:

    python tools/hd_texture_sheet.py upscale D:/GAMES/ROLLER/hd_textures_dump/tracksix --model high-fidelity
        -> stitch, upscale, cut: D:/GAMES/ROLLER/hd_textures/tracksix/000.png ...

    python tools/hd_texture_sheet.py lineup D:/GAMES/ROLLER/hd_textures_dump/tracksix:10,100,136
        -> D:/GAMES/ROLLER/hd_textures_work/lineup_tracksix.png: the chosen
           tiles from every model side by side, each repeated 2x2 the way it
           tiles in game, so invented patterns are easy to spot

Upscaled sheets are cached in hd_textures_work as <bank>_<model>.png and
reused while they're newer than the stitched sheet.

Needs Pillow only (plus Upscayl for upscale/lineup).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops

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


def stitch(bank_dir: Path, out_dir: Path | None, cols: int, gap: int) -> Path | None:
    """Writes <bank>.png and <bank>.json; returns the .json path, or None on error."""
    tiles = tile_files(bank_dir)
    if not tiles:
        print(f"No numbered PNG tiles found in {bank_dir}", file=sys.stderr)
        return None
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
            return None
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
    # Leave an identical sheet untouched so cached upscales of it stay valid.
    unchanged = False
    if sheet_path.is_file():
        with Image.open(sheet_path) as existing:
            unchanged = existing.mode == sheet.mode and existing.size == sheet.size \
                and existing.tobytes() == sheet.tobytes()
    if unchanged:
        print(f"{sheet_path.name} is unchanged")
        return layout_path
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
    return layout_path


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
        original = Image.open(source / f"{index:03d}.png").convert("RGBA")
        original_a = original.getchannel("A").resize((tile_px, tile_px), Image.NEAREST)
        if alpha_mode == "original":
            a = original_a
        else:
            # The upscaler can leave stray solid pixels on a tile's outer edge
            # where the original is see-through (cloud tiles drew a dashed
            # line down the edge of the quad). Along a band one original
            # pixel wide, keep the original's see-through pixels clear.
            band = Image.new("L", (tile_px, tile_px), 255)
            band.paste(0, (scale, scale, tile_px - scale, tile_px - scale))
            a.paste(ImageChops.darker(a, original_a), mask=band)
        a = a.point(lambda v: 255 if v >= 128 else 0)
        Image.merge("RGBA", (r, g, b, a)).save(out_dir / f"{index:03d}.png")
        written += 1

    print(f"Wrote {written} tiles at {tile_px}x{tile_px} ({scale}x) to {out_dir} "
          f"(alpha from {alpha_mode} image)")
    return 0


# --------------------------------------------------------------------------
# Upscayl automation
# --------------------------------------------------------------------------

def find_upscayl(explicit: Path | None) -> tuple[Path, Path] | None:
    """Returns (engine executable, models folder) of an Upscayl install."""
    roots = [explicit] if explicit else [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Upscayl",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Upscayl",
    ]
    for root in roots:
        for name in ("upscayl-bin.exe", "upscayl-bin"):
            engine = root / "resources" / "bin" / name
            if engine.is_file():
                return engine, root / "resources" / "models"
    print("Can't find Upscayl's engine (resources/bin/upscayl-bin); install Upscayl "
          "or pass --upscayl <install folder>", file=sys.stderr)
    return None


def available_models(models_dir: Path) -> list[str]:
    return sorted(p.stem for p in models_dir.glob("*.param"))


def resolve_model(name: str, models: list[str]) -> str | None:
    """Accepts a full model name or a short one ("high-fidelity" -> "high-fidelity-4x")."""
    for candidate in (name, f"{name}-4x"):
        if candidate in models:
            return candidate
    print(f"Unknown model {name!r}; available: {', '.join(models)}", file=sys.stderr)
    return None


# Mean per-channel difference (0-255) between an upscaled sheet shrunk back to
# the original size and the original. Working models measured 3-8 on track and
# sign sheets; a model that produced noise on this GPU (upscayl-lite) measured
# 67-104. Anything above this is treated as broken output.
BROKEN_DRIFT = 25.0


def drift_from_original(sheet: Path, upscaled: Path) -> float:
    from PIL import ImageChops, ImageStat

    with Image.open(sheet) as original, Image.open(upscaled) as big:
        small = big.convert("RGB").resize(original.size, Image.BOX)
        means = ImageStat.Stat(ImageChops.difference(small, original.convert("RGB"))).mean
    return sum(means) / len(means)


def upscaled_sheet(layout_path: Path, engine: Path, models_dir: Path, model: str) -> Path | None:
    """Upscales a stitched sheet 4x with one model, reusing a newer cached result.
    Returns None if Upscayl fails or its output looks broken."""
    sheet = layout_path.with_suffix(".png")
    out = layout_path.with_name(f"{layout_path.stem}_{model}.png")
    if out.is_file() and out.stat().st_mtime >= sheet.stat().st_mtime:
        print(f"Using cached {out.name}")
    else:
        print(f"Upscaling {sheet.name} with {model}...")
        result = subprocess.run(
            [str(engine), "-i", str(sheet), "-o", str(out), "-m", str(models_dir),
             "-n", model, "-s", "4", "-f", "png"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0 or not out.is_file():
            print(f"Upscayl failed on {sheet.name} with {model}:\n{result.stdout}{result.stderr}",
                  file=sys.stderr)
            return None
    drift = drift_from_original(sheet, out)
    if drift > BROKEN_DRIFT:
        print(f"{model} output for {sheet.name} looks broken (drift {drift:.0f}, "
              f"working models stay under {BROKEN_DRIFT:.0f}); skipping it", file=sys.stderr)
        return None
    return out


def upscale(bank_dir: Path, model_name: str, upscayl_root: Path | None,
            out_root: Path | None, cols: int, gap: int) -> int:
    found = find_upscayl(upscayl_root)
    if not found:
        return 1
    engine, models_dir = found
    model = resolve_model(model_name, available_models(models_dir))
    if not model:
        return 1
    layout_path = stitch(bank_dir, None, cols, gap)
    if not layout_path:
        return 1
    sheet = upscaled_sheet(layout_path, engine, models_dir, model)
    if not sheet:
        return 1
    return cut(layout_path, sheet, out_root, None, "auto")


def tile_from_sheet(sheet: Image.Image, layout: dict, index: int, scale: int) -> Image.Image:
    tile_px = layout["tile_size"] * scale
    x = (index % layout["cols"]) * tile_px
    y = (index // layout["cols"]) * (layout["tile_size"] + layout["gap"]) * scale
    tile = sheet.crop((x, y, x + tile_px, y + tile_px)).convert("RGBA")
    tile.putalpha(tile.getchannel("A").point(lambda v: 255 if v >= 128 else 0))
    return tile


def lineup(items: list[str], upscayl_root: Path | None, model_names: list[str] | None,
           cell: int, cols: int, gap: int) -> int:
    from PIL import ImageDraw

    found = find_upscayl(upscayl_root)
    if not found:
        return 1
    engine, models_dir = found
    models = available_models(models_dir)
    if model_names:
        models = [m for m in (resolve_model(n, models) for n in model_names) if m]
        if not models:
            return 1

    # Each item is "<bank folder>:<tiles>"; split on the last colon so drive
    # letters like D: survive.
    rows = []
    for item in items:
        folder, _, spec = item.rpartition(":")
        if not folder or not spec or not spec[0].isdigit():
            print(f"{item!r} should look like <bank folder>:<tiles>, e.g. hd_textures_dump/tracksix:10,136",
                  file=sys.stderr)
            return 1
        bank_dir = Path(folder)
        layout_path = stitch(bank_dir, None, cols, gap)
        if not layout_path:
            return 1
        layout = json.loads(layout_path.read_text())
        sheets = {}
        for model in models:
            path = upscaled_sheet(layout_path, engine, models_dir, model)
            if path:
                sheets[model] = Image.open(path)
        # Only compare models that worked for every bank in this lineup.
        models = [m for m in models if m in sheets]
        if not models:
            print("No model produced usable output", file=sys.stderr)
            return 1
        for index in parse_tiles(spec, layout["tiles"]):
            original = Image.open(bank_dir / f"{index:03d}.png").convert("RGBA")
            rows.append((f"{layout['bank']} {index:03d}", original, sheets, layout, index))

    # Grid: one row per tile, columns = original + each model. Each cell shows
    # the tile repeated 2x2 at cell/2 per copy, over a checkerboard so
    # transparency shows.
    label_w, head_h, pad = 110, 22, 6
    columns = ["original"] + models
    grid = Image.new("RGB", (label_w + len(columns) * (cell + pad), head_h + len(rows) * (cell + pad)),
                     (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for c, name in enumerate(columns):
        draw.text((label_w + c * (cell + pad) + 2, 5), name.removesuffix("-4x"), fill=(0, 0, 0))

    checker = Image.new("RGBA", (cell, cell), (200, 200, 200, 255))
    cdraw = ImageDraw.Draw(checker)
    for y in range(0, cell, 16):
        for x in range(0, cell, 16):
            if (x // 16 + y // 16) % 2:
                cdraw.rectangle([x, y, x + 15, y + 15], fill=(150, 150, 150, 255))

    half = cell // 2
    for r, (label, original, sheets, layout, index) in enumerate(rows):
        y0 = head_h + r * (cell + pad)
        draw.text((4, y0 + cell // 2 - 6), label, fill=(0, 0, 0))
        tiles = [original.resize((half, half), Image.NEAREST)]
        for model in models:
            sheet = sheets[model]
            scale = round(sheet.width / layout["width"])
            tiles.append(tile_from_sheet(sheet, layout, index, scale).resize((half, half), Image.LANCZOS))
        for c, tile in enumerate(tiles):
            block = checker.copy()
            for dy in (0, half):
                for dx in (0, half):
                    block.alpha_composite(tile, (dx, dy))
            grid.paste(block.convert("RGB"), (label_w + c * (cell + pad), y0))

    work_dir = Path(items[0].rpartition(":")[0]).resolve().parent.parent / "hd_textures_work"
    banks = "_".join(dict.fromkeys(row[0].split()[0] for row in rows))
    out = work_dir / f"lineup_{banks}.png"
    grid.save(out)
    print(f"Wrote {out} ({len(rows)} tiles x {len(columns)} columns)")
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

    p_up = sub.add_parser("upscale", help="stitch, upscale with Upscayl and cut, in one go")
    p_up.add_argument("bank_dir", type=Path, help="e.g. hd_textures_dump/tracksix")
    p_up.add_argument("--model", default="high-fidelity",
                      help="Upscayl model, e.g. high-fidelity, digital-art, upscayl-standard (default high-fidelity)")
    p_up.add_argument("--upscayl", type=Path, default=None, help="Upscayl install folder (default: found automatically)")
    p_up.add_argument("--out", type=Path, default=None,
                      help="output root (default: hd_textures next to hd_textures_dump)")
    p_up.add_argument("--cols", type=int, default=16, help="tiles per row (default 16)")
    p_up.add_argument("--gap", type=int, default=8, help="padding pixels between rows (default 8)")

    p_line = sub.add_parser("lineup", help="compare chosen tiles across every Upscayl model in one image")
    p_line.add_argument("items", nargs="+", help="<bank folder>:<tiles>, e.g. hd_textures_dump/tracksix:10,100,136")
    p_line.add_argument("--models", default=None, help="comma-separated models (default: all installed)")
    p_line.add_argument("--upscayl", type=Path, default=None, help="Upscayl install folder (default: found automatically)")
    p_line.add_argument("--cell", type=int, default=256, help="pixel size of each 2x2 cell (default 256)")
    p_line.add_argument("--cols", type=int, default=16, help="tiles per row when stitching (default 16)")
    p_line.add_argument("--gap", type=int, default=8, help="padding pixels between rows when stitching (default 8)")

    args = parser.parse_args()
    if args.command in ("stitch", "upscale", "lineup") and (args.cols < 1 or args.gap < 0):
        parser.error("--cols must be at least 1 and --gap can't be negative")
    if args.command == "stitch":
        return 0 if stitch(args.bank_dir, args.out, args.cols, args.gap) else 1
    if args.command == "upscale":
        return upscale(args.bank_dir, args.model, args.upscayl, args.out, args.cols, args.gap)
    if args.command == "lineup":
        models = args.models.split(",") if args.models else None
        return lineup(args.items, args.upscayl, models, args.cell, args.cols, args.gap)
    return cut(args.layout, args.upscaled, args.out, args.tiles, args.alpha)


if __name__ == "__main__":
    sys.exit(main())
