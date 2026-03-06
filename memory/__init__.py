from memory.mem0_adapter import Mem0Memory
from memory.zep_adapter import ZepMemory

MEMORY_SYSTEMS = {
    "mem0": Mem0Memory,
    "zep": ZepMemory,
}
