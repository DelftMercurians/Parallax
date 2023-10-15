from jaxtyping import install_import_hook


__all__ = [
    "bodies",
    "abstract_shapes",
    "collisions",
    "geometry_utils",
    "shapes",
    "design_by_contract",
]

with install_import_hook("cotix", "beartype.beartype"):
    import cotix._abstract_shapes as abstract_shapes
    import cotix._bodies as bodies
    import cotix._collisions as collisions
    import cotix._design_by_contract as design_by_contract
    import cotix._geometry_utils as geometry_utils
    import cotix._shapes as shapes
