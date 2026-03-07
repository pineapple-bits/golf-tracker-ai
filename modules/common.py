from enum import Enum
import dataclasses
from typing import Dict, List, Any, Callable


MP_INDICES = {
    'NOSE': 0, 'L_SHOULDER': 11, 'R_SHOULDER': 12, 
    'L_HIP': 23, 'R_HIP': 24, 'L_KNEE': 25, 'R_KNEE': 26, 
    'L_ANKLE': 27, 'R_ANKLE': 28, 'L_WRIST': 15, 'R_WRIST': 16
}


class Reliability(Enum):
    HIGH = "HIGH"
    MED = "MED"
    LOW = "LOW"
    INVALID = "INVALID"

class Phase(Enum):
    SETUP = "Setup / Address"
    TOP = "Top of Backswing"
    TRANSITION = "Transition"
    IMPACT = "Impact"
    FINISH = "Finish"
    TIMING = "Tempo & Sequencing"

@dataclasses.dataclass
class MetricDef:
    key: str
    name: str
    phase: Phase
    unit: str
    calc_func: Callable
    req_landmarks: List[str]
    pro_min: float = -999
    pro_max: float = 999
    is_proxy: bool = False
    desc: str = ""

@dataclasses.dataclass
class SwingEvent:
    id: int
    indices: Dict[str, int]
    type: str
    tempo: float
    cam_roll: float

@dataclasses.dataclass
class MetricResult:
    name: str
    value: Any
    unit: str
    phase: Phase
    reliability: Reliability
    flag: str = ""