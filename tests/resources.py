import pathlib
import jsonschema


def getPathToImage(filename: str) -> str:
    return str(pathlib.Path(__file__).parent.joinpath("data", filename))


_checker = jsonschema.Draft6Validator.TYPE_CHECKER.redefine(
    "array", lambda checker, value: isinstance(value, tuple))
DRAFT_VALIDATOR = jsonschema.validators.extend(jsonschema.Draft6Validator, type_checker=_checker)

CLEAN_ONE_FACE = getPathToImage("girl_front_face.jpg")
ONE_FACE = getPathToImage("one_face.jpg")
SEVERAL_FACES = getPathToImage("several_faces.jpg")
MANY_FACES = getPathToImage("many_faces.jpg")
NO_FACES = getPathToImage("kand.jpg")
