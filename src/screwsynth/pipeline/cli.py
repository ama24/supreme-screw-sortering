"""screwsynth-dataset: build a whole dataset from a recipe (catalogue -> parts -> rest poses ->
render -> post-process -> labels), from the terminal or from the GUI."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from screwsynth.common.log import setup_logging

from .recipe import STEPS, Recipe, RecipeError, list_recipes, load_recipe, recipes_dir
from .runner import Runner
from .steps import build_plan, describe, preflight, read_status

log = logging.getLogger("screwsynth.pipeline")


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _steps_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    steps = [s.strip() for s in value.split(",") if s.strip()]
    bad = set(steps) - set(STEPS)
    if bad:
        raise SystemExit(f"unknown steps {sorted(bad)}; choose from {', '.join(STEPS)}")
    return steps


def _plan(args: argparse.Namespace):
    recipe = load_recipe(args.recipe)
    steps = _steps_arg(getattr(args, "steps", None))
    if steps:
        recipe.pipeline.steps = [s for s in STEPS if s in steps]
    return build_plan(recipe)


def _print_checks(checks) -> bool:
    for c in checks:
        _out(f"[{c.level}] {c.message}")
    return not any(c.level == "error" for c in checks)


def cmd_run(args: argparse.Namespace) -> int:
    plan = _plan(args)
    if not _print_checks(preflight(plan)):
        return 2
    runner = Runner(plan, on_line=_out)
    return 0 if runner.run(fresh=args.fresh) else 1


def cmd_plan(args: argparse.Namespace) -> int:
    plan = _plan(args)
    _out(describe(plan))
    _out()
    return 0 if _print_checks(preflight(plan)) else 2


def cmd_new(args: argparse.Namespace) -> int:
    path = recipes_dir() / f"{args.name}.yaml"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists (use --overwrite)")
    if args.copy_from:
        recipe = load_recipe(args.copy_from)
        recipe.name, recipe.path = args.name, None
    else:
        recipe = Recipe(args.name)
    if args.preview:
        recipe.pipeline.preview, recipe.pipeline.scenes = True, 10
    _out(f"wrote {recipe.save(path)}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    paths = list_recipes()
    if not paths:
        _out(f"no recipes in {recipes_dir()} (create one with `screwsynth-dataset new NAME`)")
    for path in paths:
        try:
            r = load_recipe(path)
        except RecipeError as exc:
            _out(f"{path.stem:24s} INVALID: {exc}")
            continue
        st = read_status(r.run_dir())
        state = st.get("state", "never run")
        p = r.pipeline
        mode = "preview" if p.preview else "full"
        _out(f"{r.name:24s} {p.scenes:>6d} scenes  {mode:7s} {p.camera:11s} last run: {state}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import main as gui_main
    except ImportError as exc:
        raise SystemExit(f'the GUI needs NiceGUI: pip install -e ".[gui]" ({exc})') from None
    gui_main(port=args.port, show=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="screwsynth-dataset", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    steps_help = f"comma list of {','.join(STEPS)} (default: the recipe's steps)"

    sp = sub.add_parser("run", help="run a recipe (only missing outputs are made)")
    sp.add_argument("recipe", help="recipes/NAME.yaml, or just NAME")
    sp.add_argument("--steps", help=steps_help)
    sp.add_argument(
        "--fresh",
        action="store_true",
        help="delete this dataset's previous scenes and labels first",
    )
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("plan", help="show every command a run would execute, and the checks")
    sp.add_argument("recipe")
    sp.add_argument("--steps", help=steps_help)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("new", help="create recipes/NAME.yaml")
    sp.add_argument("name")
    sp.add_argument("--from", dest="copy_from", help="copy an existing recipe")
    sp.add_argument("--preview", action="store_true", help="10 low-quality scenes, a quick check")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_new)

    sp = sub.add_parser("list", help="recipes and the state of their last run")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("gui", help="open the dataset builder in the browser")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    sp.set_defaults(func=cmd_gui)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    try:
        return int(args.func(args))
    except RecipeError as exc:
        raise SystemExit(f"recipe error: {exc}") from None


def gui_entry() -> int:
    """screwsynth-gui: shortcut for `screwsynth-dataset gui`."""
    return main(["gui", *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
