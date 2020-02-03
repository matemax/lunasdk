import jsonschema

_checker = jsonschema.Draft6Validator.TYPE_CHECKER.redefine("array", lambda checker, value: isinstance(value, tuple))
jsonValidator = jsonschema.validators.extend(jsonschema.Draft6Validator, type_checker=_checker)

FLOAT = {"type": "number"}
COORDINATE_TYPE = {"type": "number", "minimum": 0}
LANDMARKS_ITEM = {"type": "array", "maxItems": 2, "minItems": 2, "items": FLOAT}

TYPE_SCORE = {"type": "number", "minimum": 0, "maximum": 1, "exclusiveMaximum": True}

TYPE_RECT = {
    "type": "object",
    "properties": {"x": COORDINATE_TYPE, "y": COORDINATE_TYPE, "width": COORDINATE_TYPE, "height": COORDINATE_TYPE},
    "required": ["x", "y", "width", "height"],
    "additionalProperties": False,
}

LANDMARKS5 = {"type": "array", "maxItems": 5, "minItems": 5, "items": LANDMARKS_ITEM}

LANDMARKS68 = {"type": "array", "maxItems": 68, "minItems": 68, "items": LANDMARKS_ITEM}

REQUIRED_FACE_DETECTION = {
    "type": "object",
    "properties": {"rect": TYPE_RECT, "score": TYPE_SCORE, "landmarks5": LANDMARKS5, "landmarks68": LANDMARKS68},
    "additionalProperties": False,
    "required": ["rect", "score"],
}

QUALITY = {
    "type": "object",
    "properties": {
        "blurriness": TYPE_SCORE,
        "dark": TYPE_SCORE,
        "illumination": TYPE_SCORE,
        "specularity": TYPE_SCORE,
        "light": TYPE_SCORE,
    },
    "additionalProperties": False,
    "required": ["blurriness", "dark", "illumination", "specularity", "light"],
}
