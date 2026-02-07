from .ABL import ABL
from .AutoEncoderDefense import AutoEncoderDefense
from .ShrinkPad import ShrinkPad
from .FineTuning import FineTuning
from .NAD import NAD
from .Pruning import Pruning
from .CutMix import CutMix
from .IBD_PSC import IBD_PSC
from .SCALE_UP import SCALE_UP
from .REFINE import REFINE
from .FLARE import FLARE

# 可选导入：MCR需要curves依赖，如果缺失则跳过
try:
    from .MCR import MCR
    _MCR_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] MCR防御方法不可用，缺少依赖: {e}")
    print("[INFO] 如需使用MCR，请安装curves依赖。其他防御方法不受影响。")
    MCR = None
    _MCR_AVAILABLE = False

# 动态构建__all__列表
__all__ = [
    'AutoEncoderDefense', 'ShrinkPad', 'FineTuning', 'NAD', 'Pruning', 'ABL', 'CutMix', 'IBD_PSC', 'SCALE_UP', 'REFINE', 'FLARE'
]

if _MCR_AVAILABLE:
    __all__.append('MCR')
