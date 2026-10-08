def intermediate_indices(depth: int) -> tuple[int, int, int, int]:
    return depth // 3 - 1, depth // 2 - 1, 2 * depth // 3 - 1, depth - 1
