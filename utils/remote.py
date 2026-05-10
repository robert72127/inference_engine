import inspect
import multiprocessing as mp
import traceback
import uuid

import zmq

from utils.logger import Logger

def _serve_object(cls, init_args, init_kwargs, endpoint):
    Logger.setup(name=f"{cls.__module__}.{cls.__name__}.worker")
    Logger.info("starting worker for %s on %s", cls.__name__, endpoint)
    obj = cls(*init_args, **init_kwargs)
    
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(endpoint)

    while True:
        msg = sock.recv_pyobj()
        try:
            op = msg["op"]
            Logger.debug("received op=%s for %s", op, cls.__name__)
            match op:
                case "call":
                    method_name = msg["method"]
                    Logger.info("calling %s.%s", cls.__name__, method_name)
                    method = getattr(obj, method_name)
                    result = method(*msg["args"], **msg["kwargs"])
                    sock.send_pyobj({"ok": True, "result": result,})
                case "shutdown":
                    Logger.info("shutting down worker for %s", cls.__name__)
                    sock.send_pyobj({"ok": True})
                    break
                case _:
                    Logger.warning("received unknown op=%s for %s", op, cls.__name__)
                    sock.send_pyobj({"ok": False,"error": f"unknown op: {op}",})
        except Exception:
            Logger.exception("worker request failed for %s", cls.__name__)
            sock.send_pyobj({"ok": False, "error": traceback.format_exc(),})

    sock.close()
    ctx.term()
    Logger.info("worker for %s stopped", cls.__name__)

def _make_remote_method(method_name):

    def remote_method(self, *args, **kwargs):
        self._sock.send_pyobj({"op": "call", "method": method_name,"args": args,"kwargs": kwargs,})
        reply = self._sock.recv_pyobj()
        if not reply["ok"]: raise RuntimeError(reply["error"])
        return reply["result"]

    return remote_method


def _wrap_subprocess_worker(original_cls, default_endpoint=None):

    class RemoteProxy:
        __remote_original_cls__ = original_cls
        __remote_default_endpoint__ = default_endpoint

        def __init__(self, *args, endpoint=None, **kwargs):
            self._endpoint = endpoint or default_endpoint or f"ipc:///tmp/{original_cls.__name__}-{uuid.uuid4().hex}.ipc"

            self._process = mp.Process(target=_serve_object, args=(original_cls, args, kwargs, self._endpoint), daemon=True,)
            self._process.start()
            Logger.info("spawned worker process for %s on %s", original_cls.__name__, self._endpoint)

            self._ctx = zmq.Context.instance()
            self._sock = self._ctx.socket(zmq.REQ)
            self._sock.connect(self._endpoint)

        def close(self):
            try:
                Logger.info("sending shutdown to %s on %s", original_cls.__name__, self._endpoint)
                self._sock.send_pyobj({"op": "shutdown"})
                self._sock.recv_pyobj()
            finally:
                self._sock.close()
                self._process.join(timeout=5)

                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join()

        def __enter__(self): return self

        def __exit__(self, exc_type, exc, tb): self.close()

    for name, fn in inspect.getmembers(original_cls, predicate=inspect.isfunction):
        if not name.startswith("_"):
            setattr(RemoteProxy, name, _make_remote_method(name))

    RemoteProxy.__name__ = original_cls.__name__
    RemoteProxy.__qualname__ = original_cls.__qualname__
    RemoteProxy.__module__ = original_cls.__module__

    return RemoteProxy


def subprocess_worker(cls=None, *, endpoint=None):
    if cls is None:
        return lambda decorated_cls: _wrap_subprocess_worker(decorated_cls, default_endpoint=endpoint)

    return _wrap_subprocess_worker(cls, default_endpoint=endpoint)
