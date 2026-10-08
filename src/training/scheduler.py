def stlr_lambda(total_steps: int, cut_frac: float = 0.1, ratio: float = 32.0):
    """Slanted Triangular Learning Rate lambda, chuyển từ notebook."""
    cut = max(1, int(total_steps * cut_frac))

    def schedule(step: int):
        if step < cut:
            p = step / cut
        else:
            denom = cut * (1 / cut_frac - 1)
            p = max(0.0, 1 - (step - cut) / max(1, denom))
        return (1 + p * (ratio - 1)) / ratio

    return schedule

