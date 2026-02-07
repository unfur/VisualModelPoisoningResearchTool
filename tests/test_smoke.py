def test_imports():
    """简单导入测试，确保核心模块可被导入。"""
    import src.core.backdoor_attacker  # noqa: F401
    import src.database.models  # noqa: F401
    import src.ui.cli_interface  # noqa: F401


