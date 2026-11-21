import math
from collections import Counter
import pandas as pd


def interaction_diversity(interactions: pd.DataFrame) -> dict:
    """Return observed type richness and Shannon entropy (natural log)."""
    counts = Counter(interactions["interaction_type"].astype(str))
    total = sum(counts.values())
    proportions = [count / total for count in counts.values()]
    shannon = -sum(p * math.log(p) for p in proportions if p > 0)
    return {
        "interaction_type_richness": len(counts),
        "interaction_type_shannon": shannon,
    }
