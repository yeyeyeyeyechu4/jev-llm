"""Convert the submission figures to RGB TIFF (350 dpi) for the acceptance-stage upload.

Reads every ``paper/figures/fig*_{en,zh}.png`` (including Fig S1) and writes
``paper/figures/tiff/<stem>.tif`` as RGB, LZW-compressed, 350 dpi.

Run:
    C:\\Users\\ZHOUSHENGZHE\\AppData\\Local\\Programs\\Python\\Python314\\python.exe make_tiff.py
"""
import pathlib

from PIL import Image

BASE = pathlib.Path(r"D:\Demo\proj\paper\figures")
OUT = BASE / "tiff"
DPI = 350

OUT.mkdir(exist_ok=True)

sources = sorted(list(BASE.glob("fig*_en.png")) + list(BASE.glob("fig*_zh.png")))
if not sources:
    raise SystemExit(f"no fig*_en.png / fig*_zh.png found in {BASE}")

for src in sources:
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    dst = OUT / (src.stem + ".tif")
    im.save(dst, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
    print(f"{dst.name}  mode={im.mode}  size={im.size}  dpi={im.info.get('dpi')}")

# sanity: every PNG has a TIFF
png_stems = {p.stem for p in sources}
tif_stems = {p.stem for p in OUT.glob("fig*.tif")}
missing = png_stems - tif_stems
if missing:
    raise SystemExit(f"missing TIFFs: {sorted(missing)}")
print(f"{len(sources)} TIFFs written to {OUT}")
