from typing import TypeVar, Generic, Callable, Generator

TaskResult = TypeVar("TaskResult")


class AsyncTask(Generic[TaskResult]):
    __slots__ = ["_coreTask", "_postProcessing"]

    def __init__(
        self, coreTask: "CoreAsyncTask", postProcessing: Callable[..., TaskResult]  # type: ignore # noqa: F821
    ):
        self._coreTask = coreTask
        self._postProcessing = postProcessing

    def get(self) -> TaskResult:
        res = self._coreTask.get()
        return self._postProcessing(*res)

    def __await__(self) -> Generator[None, None, TaskResult]:
        yield from self._coreTask.__await__()
        res = self._coreTask.result
        return self._postProcessing(*res)
