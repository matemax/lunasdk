from typing import TypeVar, Generic, Callable, Generator

TaskResult = TypeVar("TaskResult")

TaskResult1 = TypeVar("TaskResult1")


class AsyncTask(Generic[TaskResult]):
    __slots__ = ["coreTask", "postProcessing"]

    def __init__(
        self, coreTask: "CoreAsyncTask", postProcessing: Callable[..., TaskResult]  # type: ignore # noqa: F821
    ):
        self.coreTask = coreTask
        self.postProcessing = postProcessing

    def get(self) -> TaskResult:
        res = self.coreTask.get()
        return self.postProcessing(*res)

    def __await__(self) -> Generator[None, None, TaskResult]:
        yield from self.coreTask.__await__()
        res = self.coreTask.getResult()
        return self.postProcessing(*res)


def wrap(task: AsyncTask[TaskResult], pp: Callable[[TaskResult], TaskResult1]) -> AsyncTask[TaskResult1]:
    oldPP = task.postProcessing
    return AsyncTask(task.coreTask, postProcessing=lambda *coreRes: pp(oldPP(*coreRes)))
