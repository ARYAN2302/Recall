"""eval.baselines — comparison systems for the forgetting curve."""
from eval.baselines.mem0_runner import Mem0Baseline

# Also import from the parent baselines module
from eval.baselines import NaiveSFTBaseline, ReplaySFTBaseline, BASELINES

# Register Mem0
BASELINES["mem0"] = Mem0Baseline

__all__ = ["Mem0Baseline", "NaiveSFTBaseline", "ReplaySFTBaseline", "BASELINES"]
