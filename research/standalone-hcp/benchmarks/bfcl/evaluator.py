def correct(decision, expected, valid=True):
    return bool(valid and decision == expected)

# Exact original tool identities are mapped to synthetic IDs before inference.
# Only selection is scored. Arguments, AST argument matching and execution are excluded.
