# Local operations dashboard

The dashboard is a localhost-only control surface for the A6000 deployment. It
shows model, GPU, Docker and recent evidence-bundle status, and offers two
allowlisted actions: restart the known 14B launcher and run the bounded smoke.
It never accepts arbitrary shell commands.

Run it on the GPU host:

```bash
cd /home/f630/homePLUS/agent-data-plane
scripts/serve_a6000_dashboard.sh
```

From the Mac, create a tunnel and then open `http://127.0.0.1:8788`:

```bash
ssh -L 8788:127.0.0.1:8788 f630@100.65.162.35
```

Do not bind the dashboard to a Tailscale or public interface without adding an
authentication and authorization layer.

The dashboard restart action is pinned to `scripts/serve_a6000_model.sh`, which
defaults to the candidate adapter and injects its immutable model revision. Use
the same launcher with the explicit `base` argument only for a time-sliced
baseline evaluation, then restore `candidate` before returning the dashboard to
normal operation.
