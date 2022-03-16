"""
Eyes estimation example
"""
import asyncio
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_1
from lunavl.sdk.estimators.face_estimators.eyes import WarpWithLandmarks5


def estimateGazeDirection():
    """
    Estimate gaze direction.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image, detect68Landmarks=True)

    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    landMarks5Transformation = warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")

    gazeEstimator = faceEngine.createGazeEstimator()

    warpWithLandmarks5 = WarpWithLandmarks5(warp, landMarks5Transformation)
    pprint.pprint(gazeEstimator.estimate(warpWithLandmarks5).asDict())

    faceDetection2 = detector.detectOne(VLImage.load(filename=EXAMPLE_1), detect68Landmarks=True)
    warp2 = warper.warp(faceDetection2)
    landMarks5Transformation2 = warper.makeWarpTransformationWithLandmarks(faceDetection2, "L5")

    warpWithLandmarks5List = [
        WarpWithLandmarks5(warp, landMarks5Transformation),
        WarpWithLandmarks5(warp2, landMarks5Transformation2),
    ]
    estimations = gazeEstimator.estimateBatch(warpWithLandmarks5List)
    pprint.pprint([estimation.asDict() for estimation in estimations])


async def estimateGazeDirectionAsync():
    """
    Example of an async gaze direction estimation.

    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image, detect68Landmarks=True)

    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    landMarks5Transformation = warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")

    gazeEstimator = faceEngine.createGazeEstimator()

    warpWithLandmarks5 = WarpWithLandmarks5(warp, landMarks5Transformation)
    # async estimation
    pprint.pprint((await gazeEstimator.estimate(warpWithLandmarks5, asyncEstimate=True)).asDict())
    # run tasks and get results
    task1 = gazeEstimator.estimate(warpWithLandmarks5, asyncEstimate=True)
    task2 = gazeEstimator.estimate(warpWithLandmarks5, asyncEstimate=True)
    for task in (task1, task2):
        pprint.pprint(task.get().asDict())


if __name__ == "__main__":
    estimateGazeDirection()
    asyncio.run(estimateGazeDirectionAsync())
