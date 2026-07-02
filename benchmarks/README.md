# Benchmarks

Comparison oracles for the finale — not runtime dependencies of `arvyo`.
Nothing here is executed by the pipeline or its tests; this documents how
to run each tool manually against a list of TIC IDs.

## ExoMiner (Podman/Docker)

ExoMiner (Valizadegan et al.) ships as a container. To run it against a
list of TIC IDs:

```bash
# 1. Pull the image (check the ExoMiner repo/DockerHub page for the current tag)
podman pull docker.io/nasakepler/exominer:latest
# or: docker pull docker.io/nasakepler/exominer:latest

# 2. Prepare a TIC ID list, one per line
cat > tic_ids.txt <<EOF
1528696
348538431
466003005
EOF

# 3. Run the container, mounting the ID list and an output dir
podman run --rm \
  -v "$(pwd)/tic_ids.txt:/input/tic_ids.txt:ro" \
  -v "$(pwd)/exominer_out:/output" \
  docker.io/nasakepler/exominer:latest \
  --tic_ids /input/tic_ids.txt --output_dir /output

# 4. Predictions land in exominer_out/ (per-TIC scores/vetting summaries)
```

ExoMiner needs its own MAST/TESS data access set up per its documentation;
consult the upstream repo for exact container tags and input formats
before running this for real.

## triceratops

`triceratops` (Giacalone & Dressing) estimates false-positive probabilities
by modeling nearby-star contamination. Ten-line example for one candidate:

```python
import triceratops as tr

target = tr.target(ID=1528696, mission="TESS")
target.calc_probs(
    time=lc.time.value,       # your phase-folded or raw time array
    flux_0=lc.flux.value,     # normalized flux
    flux_err_0=lc.flux_err.value,
    P_orb=0.882,              # period_days from the .npz sample
)
print(target.probs)           # dict of scenario -> probability (NTP, EB, etc.)
```

Requires TIC/Gaia catalog access to build the nearby-star field; see the
triceratops repo for its `depth_correction` and `_get_sectors` setup.
