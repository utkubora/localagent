from . import tools as _tools  # noqa: F401  registers calculator, time, and file tools

try:
    from . import rag as _rag  # noqa: F401  registers vector-search tools if chromadb is installed
except ImportError:
    pass
