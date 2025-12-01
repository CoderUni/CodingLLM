import logging
import os
import sys
from datetime import datetime

_logger_initialized = False

class StreamTee:
    """
    Redirect a stream to both its original destination and a logger.
    echo_to_console controls whether to also write to the console.
    """
    def __init__(self, original_stream, logger, level, echo_to_console=True):
        self.original_stream = original_stream
        self.logger = logger
        self.level = level
        self._buffer = ""
        self.echo_to_console = echo_to_console

    def write(self, text):
        # Echo to console only if enabled (rank 0)
        if self.echo_to_console:
            self.original_stream.write(text)
            self.original_stream.flush()

        # Buffer and log full lines.
        self._buffer += text
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.strip():
                self.logger.log(self.level, line)

    def flush(self):
        if self.echo_to_console:
            self.original_stream.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

def setup_logger(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    capture_console: bool = True
) -> logging.Logger:
    global _logger_initialized
    if _logger_initialized:
        return logging.getLogger("train_logger")

    # Detect rank from torchrun env
    rank = int(os.environ.get("RANK", "0"))
    is_main = (rank == 0)

    logger = logging.getLogger("train_logger")
    logger.setLevel(log_level)
    logger.propagate = False

    # Console handler only on rank 0
    if is_main:
        console_handler = logging.StreamHandler(sys.__stdout__)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    # File per rank
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}_rank{rank}.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    # Capture stdout/stderr
    if capture_console:
        capture_logger = logging.getLogger("capture_logger")
        capture_logger.setLevel(log_level)
        capture_logger.propagate = False
        capture_logger.addHandler(file_handler)

        sys.stdout = StreamTee(sys.__stdout__, capture_logger, logging.INFO, echo_to_console=is_main)
        sys.stderr = StreamTee(sys.__stderr__, capture_logger, logging.ERROR, echo_to_console=is_main)

    _logger_initialized = True
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def get_logger() -> logging.Logger:
    """
    Retrieves the singleton logger instance.
    Initializes it with default settings if it hasn't been set up yet.
    """
    if not _logger_initialized:
        return setup_logger()
    return logging.getLogger("train_logger")