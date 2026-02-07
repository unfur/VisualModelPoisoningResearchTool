from setuptools import setup, find_packages


setup(
    name="VisualModelPoisoningResearchTool",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0,<2.0.0",
        "torchvision>=0.10.0,<0.15.0", 
        "opencv-python>=4.5.0,<5.0.0",
        "numpy>=1.21.0,<1.25.0",
        "Pillow>=8.3.0,<11.0.0",
        "lpips>=0.1.4,<0.2.0",
        "imageio>=2.9.0,<3.0.0",
        "SQLAlchemy>=1.4.0,<2.0.0",
        "PyYAML>=5.4.0,<7.0.0",
        "Flask>=2.0.0,<3.0.0",
        "rich>=10.0.0,<14.0.0",
        "tqdm>=4.62.0,<5.0.0",
    ],
    python_requires=">=3.8",
    description="基于 BackdoorBox 的视觉大模型数据中毒攻击研究工具（仅限学术与安全研究）",
    author="Researcher",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
)

