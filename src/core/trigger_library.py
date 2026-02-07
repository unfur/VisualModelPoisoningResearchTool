from typing import Any, Dict


class TriggerLibrary:
    """触发器工厂封装 BackdoorBox 的触发器接口。

    这里不直接强依赖 BackdoorBox 的内部 API，而是通过配置和动态导入在运行时绑定。
    """

    def __init__(self, backdoorbox_base: str, trigger_config: Dict[str, Any]):
        self.base_path = backdoorbox_base
        self.trigger_config = trigger_config or {}

    def create_trigger(self, trigger_type: str, **kwargs):
        """根据 trigger_type 返回具体触发器实例。

        真实实现中应从 BackdoorBox 导入对应的触发器类：
        如 backdoorbox.attacks.badnets.BadNets 之类。
        这里提供一个占位接口，方便后续进一步对接。
        """
        config = self.trigger_config.get(trigger_type, {})
        merged = {**config, **kwargs}
        # TODO: 根据 BackdoorBox 实际 API 实现
        return {"type": trigger_type, "config": merged}


