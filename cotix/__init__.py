from jaxtyping import install_import_hook


__all__ = [
    "bodies",
    "abstract_shapes",
    "collisions",
    "geometry_utils",
    "convex_shapes",
    "design_by_contract",
    "universal_shape",
]

with install_import_hook("cotix", "beartype.beartype"):
    import cotix._abstract_shapes as abstract_shapes
    import cotix._bodies as bodies
    import cotix._collisions as collisions
    import cotix._convex_shapes as convex_shapes
    import cotix._design_by_contract as design_by_contract
    import cotix._geometry_utils as geometry_utils
    import cotix._universal_shape as universal_shape
