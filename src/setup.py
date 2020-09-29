import platform
from setuptools import find_packages, setup, find_namespace_packages


def torch_urls(version):
    platform_system = platform.system()
    if platform_system == "Windows":
        return f"torch@https://download.pytorch.org/whl/cu90/torch-{version}-cp36-cp36m-win_amd64.whl#"
    return f"torch>={version}"


setup(
    name="medico",
    version="v0.0.1",
    description="MediaEval 2020, medico challenge",
    author="Nguyen Tien Phat",
    url="https://github.com/ngTienPhat/medico",
    packages=find_namespace_packages(
        exclude=["data", "result", "notebook", "script"]
    ),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires=["numpy", "Pillow", "opencv-python", "tqdm", "yacs", "tensorboardX",],
)