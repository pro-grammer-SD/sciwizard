"""Qt workers for non-blocking background execution."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThread, Signal, Slot


class WorkerSignals(QObject):
    """Signals emitted by Worker/Runnable.

    Attributes:
        started: Emitted when work begins.
        finished: Emitted on success with the result.
        error: Emitted on failure with (exception, traceback string).
        progress: Emitted with (current, total) for long tasks.
    """

    started = Signal()
    finished = Signal(object)
    error = Signal(Exception, str)
    progress = Signal(int, int)


class Worker(QRunnable):
    """A QRunnable that calls a function in a thread-pool thread.

    Args:
        fn: The function to run.
        *args: Positional arguments for fn.
        **kwargs: Keyword arguments for fn.

    Usage::

        worker = Worker(my_function, arg1, kwarg=val)
        worker.signals.finished.connect(on_done)
        worker.signals.error.connect(on_error)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:
            tb = traceback.format_exc()
            self.signals.error.emit(exc, tb)


class LongWorker(QThread):
    """A QThread subclass for tasks that need progress reporting.

    Subclass this and override :meth:`work`. Call
    ``self.progress.emit(current, total)`` from inside :meth:`work` to
    update a progress bar.
    """

    started_signal = Signal()
    finished_signal = Signal(object)
    error_signal = Signal(Exception, str)
    progress = Signal(int, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._result: Any = None

    def work(self) -> Any:
        """Override this method with the long-running task.

        Returns:
            Any result to be emitted via finished_signal.
        """
        raise NotImplementedError

    def run(self) -> None:
        self.started_signal.emit()
        try:
            result = self.work()
            self.finished_signal.emit(result)
        except Exception as exc:
            tb = traceback.format_exc()
            self.error_signal.emit(exc, tb)
