"""Physical-neighbor P2P selection, without a TPU/JAX dependency."""
import ast
import itertools
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT


def select(coords, topology):
    tree = ast.parse((TARGET / "src/ici/traffic.py").read_text())
    names = {"neighboring_p2p_chiplet_pairs", "torus_axis_distances"}
    selected = ast.Module(body=[node for node in tree.body
                               if isinstance(node, ast.FunctionDef) and node.name in names],
                          type_ignores=[])
    scope = {}
    exec(compile(selected, "traffic.py", "exec"), scope)
    return scope["neighboring_p2p_chiplet_pairs"](coords, topology)


@pytest.mark.parametrize("topology,cross_count", [((1, 1, 1), 0), ((1, 1, 2), 2),
                                                 ((2, 2, 4), 56), ((4, 4, 4), 288)])
def test_all_die_links_and_only_unique_adjacent_core_zero_links(topology, cross_count):
    coords = list(itertools.product(*(range(n) for n in topology)))
    pairs = select(coords, topology)
    dies = {(2*i, 2*i+1) for i in range(len(coords))}
    dies |= {(b, a) for a, b in dies}
    assert set(pair for pair in pairs if pair[0]//2 == pair[1]//2) == dies
    cross = [pair for pair in pairs if pair not in dies]
    assert len(cross) == cross_count
    assert len(pairs) == len(set(pairs))
    assert all(a != b for a, b in pairs)
    assert all(a % 2 == b % 2 == 0 for a, b in cross)
    assert all(sum(a == 2*i for a, _ in cross) <= 6 for i in range(len(coords)))
    assert all((b, a) in pairs for a, b in pairs)


@pytest.mark.parametrize("z_start", [0, 1, 2])
def test_three_overlapping_blocks_have_16_die_and_24_neighbor_cases(z_start):
    coords = list(itertools.product(range(2), range(2), range(z_start, z_start+2)))
    assert len(select(coords, (2, 2, 4))) == 40


def test_neither_sub_block_nor_full_slice_has_wrap_links():
    coords = [(0, 0, z) for z in range(3)]
    pairs = select(coords, (1, 1, 4))
    assert (0, 4) not in pairs  # z0 and z2 are not adjacent in full Slice.
    assert (0, 6) not in select(coords + [(0, 0, 3)], (1, 1, 4))


def test_2x2x4_boundary_and_interior_neighbor_counts():
    coords = list(itertools.product(range(2), range(2), range(4)))
    pairs = select(coords, (2, 2, 4))
    assert len(pairs) == 88  # 32 directed die links + 56 directed chip links.
    for index, (_, _, z) in enumerate(coords):
        cross = [(a, b) for a, b in pairs if a == 2*index and b//2 != index]
        assert len(cross) == (3 if z in (0, 3) else 4)


def test_larger_slice_neighbors_require_confirmed_physical_adjacency():
    with pytest.raises(ValueError, match="physical adjacency is not configured"):
        select([(0, 0, 0), (0, 0, 7)], (4, 4, 8))
