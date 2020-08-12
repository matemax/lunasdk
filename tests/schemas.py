import jsonschema

_checker = jsonschema.Draft6Validator.TYPE_CHECKER.redefine("array", lambda checker, value: isinstance(value, tuple))
jsonValidator = jsonschema.validators.extend(jsonschema.Draft6Validator, type_checker=_checker)

FLOAT = {"type": "number"}
INT = {"type": "integer"}
COORDINATE_TYPE = {"type": "number", "minimum": 0}
LANDMARKS_ITEM = {"type": "array", "maxItems": 2, "minItems": 2, "items": INT}

TYPE_SCORE = {"type": "number", "minimum": 0, "maximum": 1}

LANDMARK_WITH_SCORE_ITEM = {
    "type": "object",
    "properties": {"score": TYPE_SCORE, "point": LANDMARKS_ITEM},
    "required": ["score", "point"],
    "additionalProperties": False,
}

TYPE_RECT = {
    "type": "object",
    "properties": {"x": COORDINATE_TYPE, "y": COORDINATE_TYPE, "width": COORDINATE_TYPE, "height": COORDINATE_TYPE},
    "required": ["x", "y", "width", "height"],
    "additionalProperties": False,
}

LANDMARKS5 = {"type": "array", "maxItems": 5, "minItems": 5, "items": LANDMARKS_ITEM}

LANDMARKS68 = {"type": "array", "maxItems": 68, "minItems": 68, "items": LANDMARKS_ITEM}

LANDMARKS17 = {"type": "array", "maxItems": 17, "minItems": 17, "items": LANDMARK_WITH_SCORE_ITEM}

REQUIRED_FACE_DETECTION = {
    "type": "object",
    "properties": {"rect": TYPE_RECT, "score": TYPE_SCORE, "landmarks5": LANDMARKS5, "landmarks68": LANDMARKS68},
    "additionalProperties": False,
    "required": ["rect", "score"],
}

REQUIRED_HUMAN_BODY_DETECTION = {
    "type": "object",
    "properties": {"rect": TYPE_RECT, "score": TYPE_SCORE, "landmarks17": LANDMARKS17},
    "additionalProperties": False,
    "required": ["rect", "score"],
}

MOUTH_STATES_SCHEMA = {
    "type": "object",
    "properties": {"score": TYPE_SCORE, "occluded": TYPE_SCORE, "smile": TYPE_SCORE},
    "additionalProperties": False,
    "required": ["score", "occluded", "smile"],
}

QUALITY_SCHEMA = {
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

MASK_SCHEMA = {
    "type": "object",
    "properties": {
        "mask_in_place": TYPE_SCORE,
        "mask_not_in_place": TYPE_SCORE,
        "no_mask": TYPE_SCORE,
        "occluded_face": TYPE_SCORE,
    },
    "additionalProperties": False,
    "required": [
        "mask_in_place",
        "mask_not_in_place",
        "no_mask",
        "occluded_face"
    ],
}
