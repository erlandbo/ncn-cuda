from typing import Tuple

def group_pos_to_index(N: int, block_size: int, stage: int,
                       group_nr: int, pos_in_group: int,
                       one_based: bool = True) -> int:
    """
    Map (group_nr, pos_in_group) -> original index (1..N) for stage_groups_local_first.
    - N: total indices
    - block_size: size of each group
    - stage: stage number (0 -> local)
    - group_nr: group number (0-based if one_based=False, otherwise 1-based)
    - pos_in_group: position inside group (0-based if one_based=False, otherwise 1-based)
    - one_based: True means group_nr and pos_in_group are 1-based; returned index is 1-based.
    Returns: index in 1..N
    """
    assert N % block_size == 0, "block_size must divide N"
    m = N // block_size
    if one_based:
        b = group_nr - 1
        o = pos_in_group - 1
    else:
        b = group_nr
        o = pos_in_group

    assert 0 <= b < m, "group_nr out of range"
    assert 0 <= o < block_size, "pos_in_group out of range"

    s = 0 if stage == 0 else (1 << (stage - 1)) % m
    block = (b + o * s) % m
    idx = block * block_size + o + 1
    return idx


def index_to_group_pos(N: int, block_size: int, stage: int,
                       index: int, one_based: bool = True) -> Tuple[int,int]:
    """
    Inverse mapping: original index -> (group_nr, pos_in_group).
    Returns pair with same 1-based/0-based convention as group_pos_to_index.
    """
    assert 1 <= index <= N, "index out of range"
    assert N % block_size == 0, "block_size must divide N"
    m = N // block_size
    idx0 = index - 1
    block = idx0 // block_size
    o = idx0 % block_size

    s = 0 if stage == 0 else (1 << (stage - 1)) % m
    # group base b satisfies block = (b + o*s) % m => b = (block - o*s) mod m
    b = (block - (o * s) ) % m

    if one_based:
        return (b + 1, o + 1)
    else:
        return (b, o)


# parameters
N = 32
block_size = 8
stage = 1     # stride = 2**(stage-1) = 2

# map group 3, position 5 (1-based) -> original index
idx = group_pos_to_index(N, block_size, stage, group_nr=1, pos_in_group=2, one_based=True)
print(idx)

# inverse: which group and position contains original index 17?
grp, pos = index_to_group_pos(N, block_size, stage, index=2, one_based=True)
print(grp, pos)
