"""Archived Qwen observation schema and parser, unchanged."""
import json
import re

ATTRIBUTE_NAMES = [
    'cover_open_water', 'cover_woody_vegetation', 'cover_low_vegetation',
    'cover_bare_mineral', 'cover_impervious', 'cover_snow_ice',
    'shape_linear_network', 'shape_repeated_regular_units', 'shape_composite_site',
    'function_agriculture', 'function_residential', 'function_transport',
    'function_recreation_public_space', 'function_industrial',
    'management_water_supply', 'management_vegetation', 'woody_tree_height_form',
    'spatial_context_important',
]

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result

def parse_values(raw):
    """Accept exactly one object, optionally enclosed by one JSON code fence."""
    text = raw.strip()
    match = re.fullmatch(r'```json[ \t]*\r?\n([\s\S]*?)\r?\n```', text, re.I)
    wrapped = bool(match and text.count('```') == 2)
    obj = json.loads(match.group(1) if wrapped else text, object_pairs_hook=no_duplicates)
    require(type(obj) is dict and set(obj) == set(ATTRIBUTE_NAMES),
            'Exactly the 18 named attribute fields are required')
    values = [obj[key] for key in ATTRIBUTE_NAMES]
    require(all(v is None or (type(v) is int and v in (0, 1)) for v in values),
            'Attribute values must be integer 0, integer 1, or null')
    return values, wrapped

def normalize(raw, ended_eos=True, hit_token_limit=False):
    try:
        raw_values, wrapped = parse_values(raw)
        error = None
    except (ValueError, TypeError, RuntimeError) as exc:
        raw_values, wrapped, error = None, False, str(exc)
    valid = raw_values is not None and ended_eos and not hit_token_limit
    reason = ('schema_invalid' if raw_values is None else
              'truncated' if hit_token_limit else 'not_eos_terminated' if not ended_eos else None)
    values = raw_values[:9] + [None] * 9 if valid else [None] * 18
    return dict(values=values, known_mask=[int(v is not None) for v in values],
                raw_model_values=raw_values, schema_valid=raw_values is not None,
                description_valid=valid, wrapper_removed=wrapped, parse_error=error,
                invalid_reason=reason, fallback_used=not valid)
