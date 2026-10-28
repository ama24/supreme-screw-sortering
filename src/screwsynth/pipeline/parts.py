"""Part selection for a recipe: which classes (family + size) a dataset contains.

The GUI ticks classes; this module turns the ticks into catalogue.extra_rules (family- and
class-level enable/disable rules on top of catalogue.yaml) and back. Finer hand-written rules (part
ids, globs such as DIN933_M*x7?) are kept and applied after the generated ones.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from screwsynth.catalogue.schema import Catalogue, catalogue_from_config
from screwsynth.common.config import Config


@dataclass
class FamilyView:
    name: str
    title: str
    kind: str
    classes: list[str]  # in class order
    parts: dict[str, int]  # class -> number of parts (lengths), enabled or not
    lengths: dict[str, list[float]] = field(default_factory=dict)  # class -> lengths


@dataclass
class Overview:
    families: list[FamilyView]
    base_enabled: dict[str, bool]  # class -> enabled by catalogue.yaml alone
    finishes: dict[str, str]  # finish id -> label
    class_family: dict[str, str]

    @property
    def classes(self) -> list[str]:
        return [c for f in self.families for c in f.classes]


def overview(cfg: dict[str, Any]) -> Overview:
    """Every class of the catalogue, grouped by family, with catalogue.yaml's own selection."""
    base_cfg = Config({**cfg, "catalogue": {**(cfg.get("catalogue") or {}), "extra_rules": []}})
    cat = catalogue_from_config(base_cfg)
    fams: dict[str, FamilyView] = {}
    base: dict[str, bool] = defaultdict(bool)
    order = {c: i for i, c in enumerate(cat.classes(enabled_only=False))}
    for p in cat.parts:
        fam = cat.families[p.family]
        fv = fams.setdefault(p.family, FamilyView(p.family, fam.title, fam.kind, [], {}))
        if p.class_name not in fv.parts:
            fv.classes.append(p.class_name)
            fv.parts[p.class_name] = 0
            fv.lengths[p.class_name] = []
        fv.parts[p.class_name] += 1
        if p.length is not None:
            fv.lengths[p.class_name].append(float(p.length))
        base[p.class_name] |= p.enabled
    for fv in fams.values():
        fv.classes.sort(key=lambda c: order.get(c, 0))
    families = [fams[name] for name in cat.families if name in fams]  # catalogue.yaml order
    return Overview(
        families=families,
        base_enabled=dict(base),
        finishes={k: f.label for k, f in cat.finishes.items()},
        class_family={c: f.name for f in families for c in f.classes},
    )


def selected_classes(cfg: dict[str, Any]) -> set[str]:
    """Classes enabled with the recipe's extra_rules applied."""
    cat: Catalogue = catalogue_from_config(Config(cfg))
    return {p.class_name for p in cat.enabled_parts}


def rules_for_selection(
    ov: Overview, selected: set[str], existing: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """catalogue.extra_rules that give exactly `selected` at class level.

    Whole families switch with one rule; single classes get their own. Existing rules that are not
    plain family/class names are kept after the generated ones.
    """
    rules: list[dict[str, Any]] = []
    for fam in ov.families:
        want = {c: c in selected for c in fam.classes}
        base = {c: ov.base_enabled.get(c, False) for c in fam.classes}
        if want == base:
            continue
        if all(want.values()) or not any(want.values()):
            on = all(want.values())
            rules.append({"match": fam.name, "enabled": on})
            base = dict.fromkeys(fam.classes, on)
        rules += [{"match": c, "enabled": want[c]} for c in fam.classes if want[c] != base[c]]
    plain = {f.name for f in ov.families} | set(ov.class_family)
    kept = [r for r in existing or [] if str(r.get("match")) not in plain]
    return rules + kept


def summary(cfg: dict[str, Any]) -> dict[str, int]:
    """Counts shown next to the part selection."""
    cat = catalogue_from_config(Config(cfg))
    parts = cat.enabled_parts
    return {
        "parts": len(parts),
        "classes": len({p.class_name for p in parts}),
        "families": len({p.family for p in parts}),
    }
