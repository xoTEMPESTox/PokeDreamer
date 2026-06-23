"""
RAM address constants for Pokémon Red (NTSC/US version).
Source: https://github.com/pret/pokered and community documentation.
All addresses are in WRAM (0xC000–0xDFFF range on the Game Boy).
"""

# ── Player position ──────────────────────────────────────────────────────────
PLAYER_X = 0xD362          # Current X tile coordinate on the map
PLAYER_Y = 0xD361          # Current Y tile coordinate on the map

# ── Map ───────────────────────────────────────────────────────────────────────
MAP_ID = 0xD35E            # Current map/area ID (0x00 = Pallet Town, etc.)

# ── Player state ─────────────────────────────────────────────────────────────
PLAYER_FACING = 0xC109     # Facing direction: 0=down, 4=up, 8=left, 12=right

# ── Battle ────────────────────────────────────────────────────────────────────
IN_BATTLE = 0xD057         # 0 = overworld, 1 = wild battle, 2 = trainer battle

# ── Dialog / text box ─────────────────────────────────────────────────────────
TEXT_BOX_ID = 0xC4F1       # Non-zero when a dialog/text box is open

# ── Badges ────────────────────────────────────────────────────────────────────
BADGE_FLAGS = 0xD356       # Bitmask: bit 0=Boulder, 1=Cascade, …, 7=Earth

# ── Party ─────────────────────────────────────────────────────────────────────
PARTY_SIZE = 0xD163        # Number of Pokémon in party (0–6)

# Party current HP — one address per slot (high byte of 2-byte big-endian value)
# Low byte is always HP_ADDRESSES[slot] + 1
# Source: PWhiddy memory_addresses.py + pret/pokered disassembly
HP_ADDRESSES = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]   # current HP hi
MAX_HP_ADDRESSES = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]  # max HP hi

# Party level addresses (one per slot)
LEVEL_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

# Party species IDs (one per slot, 0 = empty)
PARTY_SPECIES_ADDRESSES = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]

# ── Event flags ───────────────────────────────────────────────────────────────
EVENT_FLAGS_START = 0xD747  # First event flag byte
EVENT_FLAGS_END   = 0xD886  # Last event flag byte (inclusive)
MUSEUM_TICKET     = 0xD754  # Example event flag: S.S. Anne ticket

# ── Map name lookup (subset) ───────────────────────────────────────────────────
MAP_NAMES = {
    0x00: "Pallet Town",
    0x01: "Viridian City",
    0x02: "Pewter City",
    0x03: "Cerulean City",
    0x04: "Lavender Town",
    0x05: "Vermilion City",
    0x06: "Celadon City",
    0x07: "Fuchsia City",
    0x08: "Cinnabar Island",
    0x09: "Indigo Plateau",
    0x0C: "Route 1",
    0x0D: "Route 2",
    0x0E: "Route 3",
    0x0F: "Route 4",
    0x10: "Route 5",
    0x11: "Route 6",
    0x12: "Route 7",
    0x13: "Route 8",
    0x14: "Route 9",
    0x15: "Route 10",
    0x16: "Route 11",
    0x17: "Route 12",
    0x18: "Route 13",
    0x19: "Route 14",
    0x1A: "Route 15",
    0x1B: "Route 16",
    0x1C: "Route 17",
    0x1D: "Route 18",
    0x1E: "Route 19",
    0x1F: "Route 20",
    0x20: "Route 21",
    0x21: "Route 22",
    0x22: "Route 23",
    0x23: "Route 24",
    0x24: "Route 25",
    0xF5: "Player's House (1F)",
    0xF6: "Player's House (2F)",
    0xF7: "Rival's House",
    0xF8: "Oak's Lab",
}

FACING_NAMES = {0: "DOWN", 4: "UP", 8: "LEFT", 12: "RIGHT"}
