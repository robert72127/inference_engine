import logging
import sys


class Logger:
    _logger: logging.Logger | None = None
    _initialized: bool = False

    @classmethod
    def setup(cls, debug: bool = False, name: str = "llm_server") -> None:
        if cls._initialized:
            return

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.propagate = False

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if debug else logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        cls._logger = logger
        cls._initialized = True

    @classmethod
    def get(cls) -> logging.Logger:
        if cls._logger is None:
            cls.setup(debug=False)
        assert cls._logger is not None
        return cls._logger

    @classmethod
    def debug(cls, msg: str, *args, **kwargs) -> None:
        cls.get().debug(msg, *args, **kwargs)

    @classmethod
    def info(cls, msg: str, *args, **kwargs) -> None:
        cls.get().info(msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg: str, *args, **kwargs) -> None:
        cls.get().warning(msg, *args, **kwargs)

    @classmethod
    def error(cls, msg: str, *args, **kwargs) -> None:
        cls.get().error(msg, *args, **kwargs)

    @classmethod
    def exception(cls, msg: str, *args, **kwargs) -> None:
        cls.get().exception(msg, *args, **kwargs)