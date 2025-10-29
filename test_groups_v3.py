from typing import List



import numpy as np

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
    return idx - 1




def stage_groups_local_first(N: int, block_size: int, stage: int) -> List[List[int]]:
    """
    Produce a partition of 1..N into m = N//block_size groups of size block_size for a given stage.
    - Stage 0: strict local blocks -> [1..block_size], [block_size+1..2*block_size], ...
    - Stage l>0: stride s = 2**(l-1) across blocks (wraps mod m), mixing blocks progressively.
    Each group G_b (b=0..m-1) contains for offsets o=0..block_size-1 the index:
        idx = (block * block_size) + o + 1
    where block = (b + o * s) % m
    Requirements: N % block_size == 0
    """
    assert N % block_size == 0, "block_size must divide N"
    m = N // block_size
    # stage 0 -> s = 0 yields contiguous blocks; stage 1 -> s = 1, stage 2 -> s = 2, etc.
    s = 0 if stage == 0 else (1 << (stage - 1)) % m

    groups: List[List[int]] = []
    for b in range(m):
        g = []
        for o in range(block_size):
            block = (b + o * s) % m
            idx = block * block_size + o + 1   # convert to 1-based index
            g.append(idx)
        groups.append(g)
    return groups


import math

# Example printout
if __name__ == "__main__":
    N = 32
    block_size = 4

    stages = int(math.log(N//block_size,2))

    for stage in range(stages):   # stages 0..4
        print(f"Stage {stage}  stride={'0 (local)' if stage==0 else 1<<(stage-1)}")
        
        groups = stage_groups_local_first(N, block_size, stage)
        print(len(np.unique(groups)))

        for i, g in enumerate(groups, 1):
            print(f"  G{i:02d}: {g}")
            print("group2idx", [group_pos_to_index(N, block_size, stage, group_nr=i-1, pos_in_group=j, one_based=False) for j in range(block_size)])
        print()
