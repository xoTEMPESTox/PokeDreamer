"""Download PokeDreamer model weights and assets from the Hugging Face Hub.

Usage:
    python lab/get_weights.py --list
    python lab/get_weights.py --only wm31
    python lab/get_weights.py --only to_route2_wm31_gate_s0
    python lab/get_weights.py
    python lab/get_weights.py --force
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Repository on Hugging Face Model Hub
REPO_ID = "xxxTEMPESTxxx/PokeDreamer"

# Manifest of weights, policies, and route states to publish / download.
# Key format: logical name identifier (used for --only <name>).
MANIFEST = {
    # --- Core Models ---
    "wm31": {
        "local": "work/runs/wm31/ckpt.pt",
        "remote": "v3/runs/wm31/ckpt.pt",
        "desc": "DIAMOND diffusion world model checkpoint (114.8k steps, EDM denoiser U-Net)",
        "category": "core",
    },
    "probe_ext": {
        "local": "work/probe_ext/probe.pt",
        "remote": "v3/probe_ext/probe.pt",
        "desc": "CNN FrameProbe reading RAM labels (map ID, coordinates, battle flag) from 72x80 frame",
        "category": "core",
    },
    "upsampler": {
        "local": "work/upsampler/upsampler.pt",
        "remote": "v3/upsampler/upsampler.pt",
        "desc": "0.7M-param CNN upsampler restoring 72x80 greyscale to native 144x160 4-shade frames",
        "category": "core",
    },

    # --- 10 Best Leg Policies (5 travel legs + 5 movement legs) ---
    "to_route2_wm31_gate_s0": {
        "local": "work/agent_rl/to_route2_wm31_gate_s0/policy.pt",
        "remote": "v3/agent_rl/to_route2_wm31_gate_s0/policy.pt",
        "desc": "Travel Leg 1: Viridian City -> Route 2 (move-gate reward, seed 0)",
        "category": "legs",
    },
    "to_gate_wm31_gate_s0": {
        "local": "work/agent_rl/to_gate_wm31_gate_s0/policy.pt",
        "remote": "v3/agent_rl/to_gate_wm31_gate_s0/policy.pt",
        "desc": "Travel Leg 2: Route 2 -> Viridian Forest South Gate (move-gate reward, seed 0)",
        "category": "legs",
    },
    "forest_wm31_gate_s0": {
        "local": "work/agent_rl/forest_wm31_gate_s0/policy.pt",
        "remote": "v3/agent_rl/forest_wm31_gate_s0/policy.pt",
        "desc": "Travel Leg 3: Viridian Forest maze (move-gate reward, seed 0)",
        "category": "legs",
    },
    "to_pewter_wm31_gate_s0": {
        "local": "work/agent_rl/to_pewter_wm31_gate_s0/policy.pt",
        "remote": "v3/agent_rl/to_pewter_wm31_gate_s0/policy.pt",
        "desc": "Travel Leg 4: North Gate -> Pewter City (move-gate reward, seed 0)",
        "category": "legs",
    },
    "to_gym_wm31_gate_s0": {
        "local": "work/agent_rl/to_gym_wm31_gate_s0/policy.pt",
        "remote": "v3/agent_rl/to_gym_wm31_gate_s0/policy.pt",
        "desc": "Travel Leg 5: Pewter City -> Brock's Gym door (move-gate reward, seed 0)",
        "category": "legs",
    },
    "to_grass_wm31_gate_s0": {
        "local": "work/agent_rl/to_grass_wm31_gate_s0/policy.pt",
        "remote": "v3/agent_rl/to_grass_wm31_gate_s0/policy.pt",
        "desc": "Movement Leg 1: Viridian City -> Route 1 grinding grass patch (move-gate, seed 0)",
        "category": "legs",
    },
    "to_viridian_pc_wm31_gate_s0": {
        "local": "work/agent_rl/to_viridian_pc_wm31_gate_s0/policy.pt",
        "remote": "v3/agent_rl/to_viridian_pc_wm31_gate_s0/policy.pt",
        "desc": "Movement Leg 2: Route 1 grass -> Viridian PokeCenter (move-gate, seed 0)",
        "category": "legs",
    },
    "leave_viridian_pc_wm31_gate_s0": {
        "local": "work/agent_rl/leave_viridian_pc_wm31_gate_s0/policy.pt",
        "remote": "v3/agent_rl/leave_viridian_pc_wm31_gate_s0/policy.pt",
        "desc": "Movement Leg 3: Exit Viridian PokeCenter to street (move-gate, seed 0)",
        "category": "legs",
    },
    "to_pewter_pc_wm31_gate_it1800_s0": {
        "local": "work/agent_rl/to_pewter_pc_wm31_gate_it1800_s0/policy.pt",
        "remote": "v3/agent_rl/to_pewter_pc_wm31_gate_it1800_s0/policy.pt",
        "desc": "Movement Leg 4: Pewter City -> Pewter PokeCenter (move-gate, 1800 it, seed 0)",
        "category": "legs",
    },
    "leave_pewter_pc_wm31_gate_mc_s0": {
        "local": "work/agent_rl/leave_pewter_pc_wm31_gate_mc_s0/policy.pt",
        "remote": "v3/agent_rl/leave_pewter_pc_wm31_gate_mc_s0/policy.pt",
        "desc": "Movement Leg 5: Exit Pewter PokeCenter to street (move-gate, merged centers, seed 0)",
        "category": "legs",
    },

    # --- 3 Headline v3 Agent Navigation Policies ---
    "leave_house": {
        "local": "work/agent_rl/leave_house/policy.pt",
        "remote": "v3/agent_rl/leave_house/policy.pt",
        "desc": "Imagination RL: Red bedroom -> Pallet Town exit (it 640; 100/100 success)",
        "category": "headline_agent",
    },
    "route1": {
        "local": "work/agent_rl/route1/policy.pt",
        "remote": "v3/agent_rl/route1/policy.pt",
        "desc": "Imagination RL: Pallet Town -> Route 1 (100/100 success, 22 median presses)",
        "category": "headline_agent",
    },
    "viridian_geo_clip": {
        "local": "work/agent_rl/viridian_geo_clip/policy.pt",
        "remote": "v3/agent_rl/viridian_geo_clip/policy.pt",
        "desc": "Imagination RL: Pallet Town -> Viridian City (geodesic + clip; 100/100 success)",
        "category": "headline_agent",
    },

    # --- 10 PPO Real-Game Baselines ---
    "ppo_leave_house": {
        "local": "work/ppo_real/leave_house/policy.pt",
        "remote": "v3/agent_rl/ppo_real_leave_house/policy.pt",
        "desc": "PPO real-game baseline: leave_house (seed 42; 100/100 success)",
        "category": "ppo",
    },
    "ppo_leave_house_s1": {
        "local": "work/ppo_real/leave_house_s1/policy.pt",
        "remote": "v3/agent_rl/ppo_real_leave_house_s1/policy.pt",
        "desc": "PPO real-game baseline: leave_house (seed 1; 100/100 success)",
        "category": "ppo",
    },
    "ppo_leave_house_s2": {
        "local": "work/ppo_real/leave_house_s2/policy.pt",
        "remote": "v3/agent_rl/ppo_real_leave_house_s2/policy.pt",
        "desc": "PPO real-game baseline: leave_house (seed 2; 100/100 success)",
        "category": "ppo",
    },
    "ppo_route1": {
        "local": "work/ppo_real/route1/policy.pt",
        "remote": "v3/agent_rl/ppo_real_route1/policy.pt",
        "desc": "PPO real-game baseline: route1 (seed 42; 98/100 success)",
        "category": "ppo",
    },
    "ppo_route1_s1": {
        "local": "work/ppo_real/route1_s1/policy.pt",
        "remote": "v3/agent_rl/ppo_real_route1_s1/policy.pt",
        "desc": "PPO real-game baseline: route1 (seed 1; 100/100 success)",
        "category": "ppo",
    },
    "ppo_route1_s2": {
        "local": "work/ppo_real/route1_s2/policy.pt",
        "remote": "v3/agent_rl/ppo_real_route1_s2/policy.pt",
        "desc": "PPO real-game baseline: route1 (seed 2; 100/100 success)",
        "category": "ppo",
    },
    "ppo_viridian": {
        "local": "work/ppo_real/viridian/policy.pt",
        "remote": "v3/agent_rl/ppo_real_viridian/policy.pt",
        "desc": "PPO real-game baseline: viridian (Manhattan reward, seed 42; 100/100 success)",
        "category": "ppo",
    },
    "ppo_viridian_s1": {
        "local": "work/ppo_real/viridian_s1/policy.pt",
        "remote": "v3/agent_rl/ppo_real_viridian_s1/policy.pt",
        "desc": "PPO real-game baseline: viridian (seed 1; 100/100 success)",
        "category": "ppo",
    },
    "ppo_viridian_s2": {
        "local": "work/ppo_real/viridian_s2/policy.pt",
        "remote": "v3/agent_rl/ppo_real_viridian_s2/policy.pt",
        "desc": "PPO real-game baseline: viridian (seed 2; 99/100 success)",
        "category": "ppo",
    },
    "ppo_viridian_geo": {
        "local": "work/ppo_real/viridian_geo/policy.pt",
        "remote": "v3/agent_rl/ppo_real_viridian_geo/policy.pt",
        "desc": "PPO real-game baseline: viridian (geodesic reward, seed 42; 95/100 success)",
        "category": "ppo",
    },

    # --- Route Save States Index ---
    "route_states_index": {
        "local": "work/route_states/index.json",
        "remote": "v3/route_states/index.json",
        "desc": "Index metadata for route save states (seed, map_id, x, y, level, hp)",
        "category": "route_states",
    },
}

# Add all 37 individual route save states to the manifest
_ROUTE_STATES = [
    "leave_pewter_pc.state",
    "leave_viridian_pc.state",
    "quest_to_gym_map2_13_26.state",
    "quest_to_route2_map1_23_26.state",
    "seed100_map1.state",
    "seed100_map12.state",
    "seed100_map13.state",
    "seed100_map2.state",
    "seed100_map41.state",
    "seed100_map47.state",
    "seed100_map50.state",
    "seed100_map51.state",
    "seed100_map54.state",
    "seed100_map58.state",
    "seed101_map1.state",
    "seed101_map12.state",
    "seed101_map13.state",
    "seed101_map2.state",
    "seed101_map41.state",
    "seed101_map47.state",
    "seed101_map50.state",
    "seed101_map51.state",
    "seed101_map54.state",
    "seed101_map58.state",
    "seed102_map1.state",
    "seed102_map12.state",
    "seed102_map13.state",
    "seed102_map2.state",
    "seed102_map41.state",
    "seed102_map47.state",
    "seed102_map50.state",
    "seed102_map51.state",
    "seed102_map54.state",
    "seed102_map58.state",
    "to_grass.state",
    "to_pewter_pc.state",
    "to_viridian_pc.state",
]

for _st in _ROUTE_STATES:
    _key = f"state_{Path(_st).stem}"
    MANIFEST[_key] = {
        "local": f"work/route_states/{_st}",
        "remote": f"v3/route_states/{_st}",
        "desc": f"PyBoy emulator route save state: {_st}",
        "category": "route_states",
    }


def format_bytes(n: int) -> str:
    """Format bytes into human-readable string."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.2f} GiB"
    elif n >= 1024 * 1024:
        return f"{n / (1024**2):.2f} MiB"
    elif n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def list_manifest() -> None:
    """List all manifest files, remote paths, local paths, and local status."""
    print("=" * 80)
    print(f"Hugging Face Model Repository: {REPO_ID}")
    print(f"Total manifest entries: {len(MANIFEST)}")
    print("=" * 80)

    found_count = 0
    total_local_bytes = 0

    # Group by category
    categories = {
        "core": "Core Models (World Model, Probe, Upsampler)",
        "legs": "10 Best Leg Policies (5 Travel + 5 Movement)",
        "headline_agent": "3 Headline v3 Agent Policies (v3_agent.md)",
        "ppo": "10 PPO Real-Game Baselines (v3_agent.md)",
        "route_states": "Route Save States (37 .state + index.json)",
    }

    for cat, cat_title in categories.items():
        cat_entries = {k: v for k, v in MANIFEST.items() if v.get("category") == cat}
        if not cat_entries:
            continue
        print(f"\n--- {cat_title} ({len(cat_entries)} files) ---")
        for k, info in cat_entries.items():
            loc_path = Path(info["local"])
            if loc_path.is_file():
                sz = loc_path.stat().st_size
                found_count += 1
                total_local_bytes += sz
                status = f"[LOCAL: {format_bytes(sz)}]"
            else:
                status = "[MISSING LOCALLY]"
            print(f"  * {k:<34s} {status:<18s} -> {info['local']}")
            print(f"    HF Path: {info['remote']}")

    print("\n" + "=" * 80)
    print(f"Summary: {found_count} of {len(MANIFEST)} files currently exist on local disk ({format_bytes(total_local_bytes)}).")
    print("=" * 80)


def download_entry(name: str, info: dict, force: bool = False) -> tuple[bool, str]:
    """Download a single entry from Hugging Face Hub using token=False."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError

    local_path = Path(info["local"])
    remote_path = info["remote"]

    if local_path.is_file() and not force:
        sz = local_path.stat().st_size
        return True, f"SKIPPED (already exists: {format_bytes(sz)})"

    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cached_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            token=False,
        )
        shutil.copy2(cached_file, local_path)
        sz = local_path.stat().st_size

        # If it's a PPO policy, also populate the mirrored path in work/agent_rl/ if appropriate
        if "ppo_real" in str(local_path):
            mirror_dir = Path("work/agent_rl") / f"ppo_real_{local_path.parent.name}"
            mirror_dir.mkdir(parents=True, exist_ok=True)
            mirror_file = mirror_dir / "policy.pt"
            if not mirror_file.exists() or force:
                try:
                    shutil.copy2(local_path, mirror_file)
                except Exception:
                    pass

        return True, f"DOWNLOADED ({format_bytes(sz)})"

    except EntryNotFoundError:
        return False, f"MISSING ON HF: '{remote_path}' not found in repo '{REPO_ID}' (HTTP 404; not published yet)"
    except RepositoryNotFoundError:
        return False, f"REPO NOT FOUND: '{REPO_ID}' does not exist on Hugging Face"
    except HfHubHTTPError as e:
        return False, f"HTTP ERROR ({getattr(e.response, 'status_code', 'unknown')}): {e}"
    except Exception as e:
        return False, f"ERROR: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PokeDreamer weights and route states from Hugging Face Hub."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all manifest weights and local availability without downloading.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Download only a specific weight by key name, subpath, or category (e.g. 'wm31', 'legs', 'ppo').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing files.",
    )
    args = parser.parse_args()

    if args.list:
        list_manifest()
        return

    # Select target items
    if args.only:
        query = args.only.strip().lower()
        # Match exact key, category, or substring in key/path
        selected = {
            k: v
            for k, v in MANIFEST.items()
            if k.lower() == query
            or v.get("category", "").lower() == query
            or query in k.lower()
            or query in v["local"].lower()
            or query in v["remote"].lower()
        }
        if not selected:
            print(f"[ERROR] No manifest entries match '--only {args.only}'.")
            print("Run 'python lab/get_weights.py --list' to see available entries and categories.")
            sys.exit(1)
        print(f"Filtering by '--only {args.only}': matched {len(selected)} file(s).")
    else:
        selected = MANIFEST

    print(f"Target repository: https://huggingface.co/{REPO_ID}")
    print(f"Files to process: {len(selected)}\n")

    success_count = 0
    skip_count = 0
    fail_count = 0
    failures = []

    for idx, (key, info) in enumerate(selected.items(), 1):
        print(f"[{idx}/{len(selected)}] {key} -> {info['local']} ...", end=" ", flush=True)
        ok, msg = download_entry(key, info, force=args.force)
        print(msg)
        if ok:
            if "SKIPPED" in msg:
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1
            failures.append((key, info["remote"], msg))

    print("\n" + "=" * 80)
    print("Download Summary:")
    print(f"  Successfully downloaded: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Failed / Missing on HF : {fail_count}")
    print("=" * 80)

    if failures:
        print("\nMissing or failed files:")
        for k, rem, err in failures:
            print(f"  - {k} ({rem}): {err}")
        print("\nNote: The owner will perform the upload to Hugging Face before public release.")
        print("Once uploaded, running this script will download all manifest files automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()
