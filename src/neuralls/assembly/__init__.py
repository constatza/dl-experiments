"""Assembly layer: bridges configuration models to domain objects.

Modules here depend on both the configuration layer (Pydantic models, enums)
and the solver/preconditioner layer (concrete algorithm classes).  They are
explicit glue — not part of either layer — and must never be imported from
inside the solver or configuration packages.
"""
