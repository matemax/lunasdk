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

MOUTH_SMILE_TYPE_ESTIMATIONS = {
    "type": "object",
    "properties": {"regular": TYPE_SCORE, "with_teeth": TYPE_SCORE},
    "additionalProperties": False,
    "required": ["regular", "with_teeth"],
}
MOUTH_SMILE_TYPE_ENUM = {"type": "string", "enum": ["none", "with_teeth", "regular"]}

MOUTH_SMILE_TYPE = {
    "type": "object",
    "properties": {"estimations": MOUTH_SMILE_TYPE_ESTIMATIONS, "predominant_type": MOUTH_SMILE_TYPE_ENUM},
    "additionalProperties": False,
    "required": ["estimations", "predominant_type"],
}

MOUTH_PROPERTIES = {
    "type": "object",
    "properties": {
        "smile_type": MOUTH_SMILE_TYPE,
    },
    "additionalProperties": False,
    "required": ["smile_type"],
}

MOUTH_STATES_SCHEMA = {
    "type": "object",
    "properties": {"opened": TYPE_SCORE, "occluded": TYPE_SCORE, "smile": TYPE_SCORE, "properties": MOUTH_PROPERTIES},
    "additionalProperties": False,
    "required": ["opened", "occluded", "smile", "properties"],
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

MASK_ESTIMATION_SCHEMA = {
    "type": "object",
    "properties": {"medical_mask": TYPE_SCORE, "missing": TYPE_SCORE, "occluded": TYPE_SCORE},
    "additionalProperties": False,
    "required": ["medical_mask", "missing", "occluded"],
}

MASK_FACE_OCCLUSION_ESTIMATION_SCHEMA = {
    "type": "object",
    "properties": {
        "full": TYPE_SCORE,
        "clear": TYPE_SCORE,
        "correct": TYPE_SCORE,
        "partially": TYPE_SCORE,
        "mouth": TYPE_SCORE,
        "chin": TYPE_SCORE,
    },
    "additionalProperties": False,
    "required": ["full", "clear", "correct", "partially", "mouth", "chin"],
}

MASK_FACE_OCCLUSION_SCHEMA = {
    "type": "object",
    "properties": {
        "estimations": MASK_FACE_OCCLUSION_ESTIMATION_SCHEMA,
        "predominant_occlusion": {"type": "string", "enum": ["full", "clear", "correct", "partially", "mouth", "chin"]},
    },
    "required": ["estimations", "predominant_occlusion"],
    "additionalProperties": False,
}

MASK_SCHEMA = {
    "type": "object",
    "properties": {
        "estimations": MASK_ESTIMATION_SCHEMA,
        "predominant_mask": {"type": "string", "enum": ["medical_mask", "missing", "occluded"]},
        "face_occlusion": MASK_FACE_OCCLUSION_SCHEMA,
    },
    "required": ["estimations", "predominant_mask", "face_occlusion"],
    "additionalProperties": False,
}

GLASSES_SCHEMA = {
    "type": "object",
    "properties": {"glasses": {"type": "string", "enum": ["no_glasses", "eyeglasses", "sunglasses"]}},
    "required": ["glasses"],
    "additionalProperties": False,
}

LIVENESSV1_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction": {"type": "string", "enum": ["real", "spoof", "unknown"]},
        "estimations": {
            "type": "object",
            "required": ["quality", "score"],
            "properties": {"quality": TYPE_SCORE, "score": TYPE_SCORE},
            "additionalProperties": False,
        },
    },
    "required": ["prediction", "estimations"],
    "additionalProperties": False,
}

CREDIBILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "estimations": {
            "type": "object",
            "properties": {"score": TYPE_SCORE},
            "required": ["score"],
            "additionalProperties": False,
        },
        "prediction": {"type": "string", "enum": ["reliable", "non_reliable"]},
    },
    "required": ["estimations", "prediction"],
    "additionalProperties": False,
}
