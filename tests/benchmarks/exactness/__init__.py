"""CG iteration count exactness validation benchmarks.

This benchmark suite validates that our CG solver implementations produce
exactly the same iteration counts as reference implementations and theoretical
equivalents.

Benchmarks:
1. FCG(-1) vs PCG: Verifies FCG with full orthogonalization matches classical PCG
2. PCG-ours vs PCG-scipy: Verifies our PCG implementation matches scipy reference
"""
