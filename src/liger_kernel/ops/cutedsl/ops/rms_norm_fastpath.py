"""Pure-Python eligibility helpers for the CuTe DSL RMSNorm vector path."""


def fast_path_vector_width(*element_sizes: int) -> int:
    """Return the common 16-byte vector width for all participating tensors."""
    largest = max(element_sizes)
    if largest <= 0 or 16 % largest:
        raise ValueError(f"element sizes must divide 16, got {element_sizes}")
    return 16 // largest


def supports_global_cp_async(vector_width: int, *element_sizes: int) -> bool:
    """Whether every staged tensor gets a legal 16-byte cp.async.cg copy."""
    return all(vector_width * element_size == 16 for element_size in element_sizes)


def persistent_strip_count(sm_count: int, n_cols: int, n_rows: int, capability: tuple[int, int]) -> int:
    """Choose the B200/Hopper persistent-CTA count, capped to available rows."""
    major, _ = capability
    if major >= 10:  # Blackwell, including B200
        multiplier = 4 if n_cols <= 2048 else 2
    elif major == 9:  # Hopper
        multiplier = 4 if n_cols <= 2048 else 2 if n_cols <= 4096 else 1
    else:
        multiplier = 1
    return max(1, min(sm_count * multiplier, n_rows))
