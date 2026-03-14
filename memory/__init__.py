from memory.mem0_adapter import Mem0Memory
from memory.zep_adapter import GraphitiDryRunMemory

MEMORY_SYSTEMS = {
    "mem0": Mem0Memory,
    "graphiti_dryrun": GraphitiDryRunMemory,
}
