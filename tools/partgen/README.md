# 02 partgen

Parametric mesh generator (numpy + trimesh, outside Blender) that turns each catalogue part into a detailed render mesh with real helical threads, a low-poly compound collision proxy, keypoints and a part.json, with dimension and watertightness checks.

- **Inputs:** configs/catalogue.yaml
- **Outputs:** data/assets/<part_id>/{render.ply, collision.obj, part.json, preview.png}, tools/partgen/contact_sheet.png
- **Guide:** [docs/prompts/02_partgen.txt](../../docs/prompts/02_partgen.txt)

**Status: TODO**
