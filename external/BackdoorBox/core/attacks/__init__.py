from ast import Import
from .BadNets import BadNets
from .Blended import Blended
from .LabelConsistent import LabelConsistent
from .Refool import Refool
from .WaNet import WaNet
from .Blind import Blind
from .IAD import IAD
from .LIRA import LIRA
from .PhysicalBA import PhysicalBA
from .ISSBA import ISSBA
from .TUAP import TUAP
from .SleeperAgent import SleeperAgent
from .BATT import BATT
from .AdaptivePatch import AdaptivePatch

# 可选导入：BAAT需要ArtFlow依赖，如果缺失则跳过
try:
    from .BAAT import BAAT
    _BAAT_AVAILABLE = True
except ImportError as e:
    # print(f"[WARNING] BAAT攻击方法不可用，缺少依赖: {e}")
    # print("[INFO] 如需使用BAAT，请安装ArtFlow依赖。其他攻击方法不受影响。")
    BAAT = None
    _BAAT_AVAILABLE = False

# 动态构建__all__列表
__all__ = [
    'BadNets', 'Blended','Refool', 'WaNet', 'LabelConsistent', 'Blind', 'IAD', 'LIRA', 
    'PhysicalBA', 'ISSBA','TUAP', 'SleeperAgent','BATT', 'AdaptivePatch'
]

if _BAAT_AVAILABLE:
    __all__.append('BAAT')
