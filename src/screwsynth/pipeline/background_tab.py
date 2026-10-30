"""GUI tab 'Background': what the surface under the parts looks like.

Weights of the three sources and the solid colours belong to the recipe (scenegen.plate.*);
the photo library (paths.plate_dir/backgrounds.yaml) is shared by all recipes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nicegui import ui

from screwsynth.common.paths import resolve
from screwsynth.scenegen import backgrounds as bg

from .recipe import get_path

PHOTO_GUIDE = """
**How to photograph a surface (craft mat, bare table, ...)**

1. Lay it flat and photograph it **straight down**, phone parallel to the surface, from far enough
   that the photo covers **at least the 220 mm plate** (more is fine).
2. **Soft, even light**: no flash glare, no hard shadows. The renderer adds its own lighting and
   shadows, so the photo should look like the bare surface.
3. Put a **ruler** or tape measure in the photo (or use the mat's printed grid): you need a known
   length to set the scale. You can crop the ruler away afterwards, as long as you then measure the
   crop's width.
4. Resolution: at least **5 px per mm** (a 300 mm wide photo: 1500 px or more). The machine's camera
   sees about 11 px/mm, so more is better.
5. Avoid perspective: a tilted photo makes the far side smaller. If unsure, crop to the central part.

In the renders the photo is laid on the plate at its true size, turned and shifted randomly in every
scene, and mirror-tiled where it is smaller than the plate.
"""


def _hex_ok(value: str) -> bool:
    return bool(re.fullmatch(r"#?[0-9A-Fa-f]{6}", (value or "").strip()))


def background_tab(page: Any) -> None:
    ed, w = page.ed, page.w
    merged = ed.merged()
    plate_dir = resolve(get_path(merged, "paths.plate_dir", "data/plate"))
    calib_dir = resolve(get_path(merged, "paths.calibration_dir", "data/calibration"))
    have_photo = (
        calib_dir / "plate_empty.png"
    ).is_file() and ed.recipe.pipeline.camera == "calibration"

    switches: dict[str, list[Any]] = {s: [] for s in bg.SOURCES}

    def is_on() -> dict[str, bool]:
        return bg.enabled(dict(get_path(ed.merged(), "scenegen.plate", {}) or {}))

    def set_on(src: str, on: bool, notify: bool = True) -> None:
        cur = is_on()
        if cur[src] == on:
            return
        cur[src] = on
        ed.set("scenegen.plate.background_enabled", cur)
        for sw in switches[src]:  # keep every switch for this source in step
            sw.set_value(on)
        refresh_sources()
        if (
            notify
            and not any(cur[s] for s in ("images", "colours"))
            and not (have_photo and cur["calibrated_photo"])
        ):
            ui.notify(
                "Every background is switched off: the renders fall back to the solid colours",
                type="warning",
            )

    def switch(src: str, label: str = "") -> Any:
        sw = ui.switch(label, value=is_on()[src], on_change=lambda e: set_on(src, bool(e.value)))
        sw.mark(f"enable-{src}")
        switches[src].append(sw)
        return sw

    with ui.column().classes("w-full max-w-5xl gap-3"):
        # ------------------------------------------------------------------ sources
        with ui.card().classes("w-full"):
            ui.label("Sources (switch them on/off; those on share the scenes by weight)").classes(
                "font-bold"
            )
            sources_box = ui.column().classes("w-full gap-1")

        pct_labels: dict[str, Any] = {}

        def refresh_sources() -> None:
            sources_box.clear()
            with sources_box:
                draw_sources()

        def update_shares() -> None:
            n = len(bg.usable_images(plate_dir, bg.load_library(plate_dir)))
            share = bg.weights(
                dict(get_path(ed.merged(), "scenegen.plate", {}) or {}), have_photo, n > 0
            )
            total = sum(share.values()) or 1
            for s, lab in pct_labels.items():
                lab.set_text(f"{share[s] / total:.0%} of scenes")

        def draw_sources() -> None:
            weights = dict(ed.get("scenegen.plate.background_weights") or bg.DEFAULT_WEIGHTS)
            n_images = len(bg.usable_images(plate_dir, bg.load_library(plate_dir)))
            notes = {
                "calibrated_photo": (
                    "photo of your empty plate from the calibration"
                    + (
                        ""
                        if have_photo
                        else " — not available (needs the calibrated camera and plate_empty.png)"
                    )
                ),
                "images": f"your surface photos below ({n_images} usable)"
                + ("" if n_images else " — none yet, upload one below"),
                "colours": "one of the solid colours below",
            }
            on = is_on()
            for src in bg.SOURCES:
                with ui.row().classes("w-full items-center no-wrap gap-3"):
                    ui.switch(value=on[src], on_change=lambda e, s=src: set_on(s, bool(e.value)))
                    ui.label(src.replace("_", " ")).classes("w-36 font-medium")
                    val = ui.label(f"{float(weights.get(src, 0)):.2f}").classes("w-10")

                    def put(e: Any, src: str = src, val: Any = val) -> None:
                        cur = dict(
                            ed.get("scenegen.plate.background_weights") or bg.DEFAULT_WEIGHTS
                        )
                        cur[src] = round(float(e.value), 2)
                        ed.set("scenegen.plate.background_weights", cur)
                        val.set_text(f"{float(e.value):.2f}")
                        update_shares()

                    sl = ui.slider(
                        min=0, max=1, step=0.05, value=float(weights.get(src, 0)), on_change=put
                    ).classes("w-48")
                    sl.set_enabled(on[src])
                    pct_labels[src] = ui.label().classes("w-24 text-xs")
                    ui.label(notes[src]).classes("text-xs text-grey-7")
            ui.label(
                "Sources that are off or not available drop out; the others share the scenes by weight."
            ).classes("text-xs text-grey-7")
            update_shares()

        refresh_sources()

        # ------------------------------------------------------------------ colours
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center"):
                ui.label("Solid colours").classes("font-bold")
                switch("colours", "use solid colours")
            ui.label(
                "Saved in this recipe. The brightness varies a little per scene (colour jitter)."
            ).classes("text-xs text-grey-7")
            rows = ui.column().classes("w-full gap-1")

            def store(colours: list[dict[str, str]]) -> None:
                ed.set("scenegen.plate.colours", colours)

            def draw_colours() -> None:
                rows.clear()
                colours = bg.colour_list(merged_plate())
                with rows:
                    for i, c in enumerate(colours):
                        with ui.row().classes("items-center no-wrap gap-2"):
                            swatch = ui.element("div").classes("w-8 h-8 rounded border")
                            swatch.style(f"background:{c['hex']}")

                            def rename(e: Any, i: int = i) -> None:
                                cs = bg.colour_list(merged_plate())
                                cs[i]["name"] = (e.value or f"colour_{i + 1}").strip()
                                store(cs)

                            def recolour(e: Any, i: int = i, swatch: Any = swatch) -> None:
                                if not _hex_ok(e.value):
                                    return
                                cs = bg.colour_list(merged_plate())
                                cs[i]["hex"] = "#" + e.value.strip().lstrip("#").upper()
                                store(cs)
                                swatch.style(f"background:{cs[i]['hex']}")

                            def remove(i: int = i) -> None:
                                cs = bg.colour_list(merged_plate())
                                if len(cs) <= 1:
                                    ui.notify(
                                        "keep at least one colour (set the colours weight to 0 to use none)",
                                        type="warning",
                                    )
                                    return
                                del cs[i]
                                store(cs)
                                draw_colours()

                            ui.input("name", value=c["name"], on_change=rename).props(
                                "dense outlined"
                            ).classes("w-48")
                            ui.color_input(
                                "colour (#RRGGBB)", value=c["hex"], on_change=recolour
                            ).props("dense outlined").classes("w-48")
                            ui.button(icon="delete", on_click=remove).props(
                                "flat dense round"
                            ).tooltip("remove").mark(f"remove-colour-{i}")

            def merged_plate() -> dict:
                return dict(get_path(ed.merged(), "scenegen.plate", {}) or {})

            def add() -> None:
                cs = bg.colour_list(merged_plate())
                cs.append({"name": f"colour_{len(cs) + 1}", "hex": "#808080"})
                store(cs)
                draw_colours()

            draw_colours()
            with ui.row().classes("items-center gap-2"):
                ui.button("Add colour", icon="add", on_click=add).props("flat")
                ui.button(
                    "Reset to silicone blue",
                    on_click=lambda: (ed.reset("scenegen.plate.colours"), draw_colours()),
                ).props("flat")
            w.field("scenegen.plate.colour_jitter", "Colour jitter")

        # ------------------------------------------------------------------ photos
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center"):
                ui.label(f"Surface photos (shared by all recipes: {plate_dir})").classes(
                    "font-bold"
                )
                switch("images", "use surface photos")
            with ui.expansion("How to take and scale a surface photo", icon="help").classes(
                "w-full"
            ):
                ui.markdown(PHOTO_GUIDE)
            table = ui.column().classes("w-full gap-2")

            def library() -> list[bg.BackgroundImage]:
                return bg.load_library(plate_dir)

            def save(items: list[bg.BackgroundImage]) -> None:
                plate_dir.mkdir(parents=True, exist_ok=True)
                bg.save_library(plate_dir, items)

            def draw_table() -> None:
                table.clear()
                items = library()
                with table:
                    if not items:
                        ui.label("No photos yet.").classes("text-sm text-grey-7")
                    for i, im in enumerate(items):
                        path = plate_dir / im.file
                        with ui.row().classes("w-full items-center no-wrap gap-3"):
                            if path.is_file():
                                from .gui import thumb

                                ui.image(thumb(path, 240)).classes("w-28")
                                px_w = _image_size(path)
                            else:
                                ui.label("file missing").classes("w-28 text-negative text-xs")
                                px_w = None
                            with ui.column().classes("gap-0 grow"):
                                ui.label(im.file).classes("font-medium")
                                scale_txt = ui.label(_scale_text(px_w, im.width_mm)).classes(
                                    "text-xs text-grey-7"
                                )

                            def set_width(
                                e: Any, i: int = i, px_w: Any = px_w, scale_txt: Any = scale_txt
                            ) -> None:
                                if not e.value or float(e.value) <= 0:
                                    return
                                items = library()
                                items[i].width_mm = float(e.value)
                                save(items)
                                scale_txt.set_text(_scale_text(px_w, float(e.value)))

                            def set_weight(e: Any, i: int = i) -> None:
                                if e.value is None:
                                    return
                                items = library()
                                items[i].weight = float(e.value)
                                save(items)

                            def set_enabled(e: Any, i: int = i) -> None:
                                items = library()
                                items[i].enabled = bool(e.value)
                                save(items)

                            def delete(i: int = i) -> None:
                                items = library()
                                gone = items.pop(i)
                                save(items)
                                (plate_dir / gone.file).unlink(missing_ok=True)
                                draw_table()

                            ui.number(
                                "real width (mm)",
                                value=im.width_mm,
                                min=1,
                                step=1,
                                on_change=set_width,
                            ).props("dense outlined").classes("w-36")
                            if path.is_file():
                                ui.button(
                                    "Scale with a ruler",
                                    icon="straighten",
                                    on_click=lambda path=path, i=i: scale_dialog(path, i),
                                ).props("flat dense")
                            ui.number(
                                "weight", value=im.weight, min=0, step=0.5, on_change=set_weight
                            ).props("dense outlined").classes("w-24")
                            ui.switch("use", value=im.enabled, on_change=set_enabled)
                            ui.button(icon="delete", on_click=delete).props(
                                "flat dense round"
                            ).tooltip("delete photo")

            def scale_dialog(path: Path, index: int) -> None:
                """Click both ends of a known length on the ruler, type it: the width follows."""
                from PIL import Image

                with Image.open(path) as im:
                    w_px, h_px = im.size
                clicks: list[tuple[float, float]] = []
                with ui.dialog() as dlg, ui.card().classes("w-[60rem] max-w-full"):
                    ui.label(f"Scale {path.name}").classes("font-bold")
                    ui.label(
                        "Click the two ends of a known length on the ruler in the photo (e.g. the 0 and the "
                        "200 mm marks), type that length, and save."
                    ).classes("text-sm")
                    info = ui.label("no points yet").classes("text-xs text-grey-7")

                    def click(e: Any) -> None:
                        if len(clicks) == 2:
                            clicks.clear()
                        clicks.append((e.image_x, e.image_y))
                        marks = "".join(
                            f'<circle cx="{x}" cy="{y}" r="{max(w_px, h_px) / 150}" fill="red" />'
                            for x, y in clicks
                        )
                        if len(clicks) == 2:
                            (x1, y1), (x2, y2) = clicks
                            marks += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="red" stroke-width="{max(w_px, h_px) / 400}" />'
                        img_el.content = marks
                        info.set_text(
                            f"{len(clicks)} point(s)"
                            + (f": {_dist(clicks):.0f} px apart" if len(clicks) == 2 else "")
                        )

                    img_el = ui.interactive_image(
                        str(path), on_mouse=click, events=["click"], cross=True
                    ).classes("w-full")
                    dist = ui.number(
                        "length between the two points (mm)", value=100, min=1, step=1
                    ).props("dense outlined")

                    def apply() -> None:
                        if len(clicks) != 2 or not dist.value:
                            ui.notify("click two points and type their distance", type="warning")
                            return
                        width = bg.width_from_ruler(w_px, clicks[0], clicks[1], float(dist.value))
                        items = library()
                        items[index].width_mm = round(width, 1)
                        save(items)
                        ui.notify(
                            f"{path.name}: {width:.1f} mm wide ({w_px / width:.1f} px/mm)",
                            type="positive",
                        )
                        dlg.close()
                        draw_table()

                    with ui.row():
                        ui.button("Save scale", icon="check", on_click=apply)
                        ui.button("Cancel", on_click=dlg.close).props("flat")
                dlg.open()

            async def uploaded(e: Any) -> None:
                name = _safe_name(e.file.name)
                if Path(name).suffix.lower() not in bg.IMAGE_SUFFIXES:
                    ui.notify(
                        f"{name}: not an image ({', '.join(bg.IMAGE_SUFFIXES)})", type="warning"
                    )
                    return
                plate_dir.mkdir(parents=True, exist_ok=True)
                target = plate_dir / name
                await e.file.save(target)
                items = [im for im in library() if im.file != name]
                items.append(bg.BackgroundImage(name, 300.0))
                save(items)
                draw_table()
                if not is_on()["images"]:  # a new photo: use it, instead of the solid colours
                    set_on("images", True, notify=False)
                    set_on("colours", False, notify=False)
                    ui.notify(
                        "Surface photos switched on and solid colours off (switch colours back on to mix both)",
                        type="info",
                    )
                ui.notify(
                    f"{name} added: now set its real width (or scale it with a ruler)", type="info"
                )
                scale_dialog(target, len(items) - 1)

            draw_table()
            ui.upload(
                label="Upload surface photos", multiple=True, auto_upload=True, on_upload=uploaded
            ).props("accept=image/*").classes("w-full")


def _image_size(path: Path) -> tuple[int, int] | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            return im.size
    except OSError:
        return None


COVERAGE_TEXT = {
    "any angle": "covers the view at any angle (no repeats)",
    "straight only": "covers the view only when straight; turned scenes repeat it mirrored at the edges "
    "(a photo of at least 311 mm in both directions covers every angle)",
    "too small": "smaller than the view: always repeated mirrored (at least 220 mm, better 311 mm, avoids it)",
}


def _scale_text(size: tuple[int, int] | int | None, width_mm: float) -> str:
    if isinstance(size, int):
        size = (size, size)
    if not size or width_mm <= 0:
        return f"{width_mm:g} mm wide"
    w_px, h_px = size
    ppm = w_px / width_mm
    hint = "" if ppm >= 5 else " — low resolution, the camera sees ~11 px/mm"
    height = width_mm * h_px / w_px
    cover = COVERAGE_TEXT[bg.coverage(width_mm, h_px / w_px)]
    return f"{width_mm:g} x {height:.0f} mm, {w_px} px: {ppm:.1f} px/mm{hint}. It {cover}."


def _dist(clicks: list[tuple[float, float]]) -> float:
    (x1, y1), (x2, y2) = clicks
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _safe_name(name: str) -> str:
    stem, dot, ext = Path(name).name.rpartition(".")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem or ext)
    return f"{stem}.{ext.lower()}" if dot else stem
