"""
Module realize simple examples following features:
    * face landmarks estimation
    * async face landmarks estimation
"""
import asyncio

from lunavl.sdk.detectors.facedetector import FaceLandmarks
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_SEVERAL_FACES


def estimateLandmarks():
    """
    Estimate face landmarks.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()
    landmarksEstimator = faceEngine.createFaceLandmarksEstimator()

    imageWithSeveralFaces = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)

    severalHumans = detector.detect([imageWithSeveralFaces])
    estimation = landmarksEstimator.estimate(severalHumans[0][0].face, landmarksType=FaceLandmarks.Landmarks5)
    print("landmarks5", estimation.asDict())
    estimation = landmarksEstimator.estimate(severalHumans[0][0].face, landmarksType=FaceLandmarks.Landmarks68)
    print("landmarks68", estimation.asDict())
    estimations = landmarksEstimator.estimateBatch(
        [human.face for human in severalHumans[0] if human.face], landmarksType=FaceLandmarks.Landmarks5
    )
    print("landmarks5 batch", [estimation.asDict() for estimation in estimations])
    estimations = landmarksEstimator.estimateBatch(
        [human.face for human in severalHumans[0] if human.face], landmarksType=FaceLandmarks.Landmarks68
    )
    print("landmarks68 batch", [estimation.asDict() for estimation in estimations])


async def asyncEstimateLandmarks():
    """
    Async estimate face landmarks.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()
    landmarksEstimator = faceEngine.createFaceLandmarksEstimator()

    image = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    detections = await detector.detect([image], asyncEstimate=True)

    estimation = await landmarksEstimator.estimate(
        detections[0][0].face, landmarksType=FaceLandmarks.Landmarks5, asyncEstimate=True
    )
    print("landmarks5", estimation.asDict())
    faces = [humanDetection.face for humanDetection in detections[0]]
    task1 = landmarksEstimator.estimateBatch(faces, asyncEstimate=True, landmarksType=FaceLandmarks.Landmarks5)
    task2 = landmarksEstimator.estimateBatch(faces, asyncEstimate=True, landmarksType=FaceLandmarks.Landmarks68)

    estimations = task1.get()
    print("landmarks5 batch", [estimation.asDict() for estimation in estimations])
    estimations = task2.get()
    print("landmarks68 batch", [estimation.asDict() for estimation in estimations])


if __name__ == "__main__":
    estimateLandmarks()
    asyncio.run(asyncEstimateLandmarks())
