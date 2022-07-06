from pathlib import Path


def get_project_dir() -> Path:
    """
    获取项目目录
    :return:
    """
    project_dir = Path(__file__).resolve().parent.parent
    return project_dir


def get_log_dir() -> Path:
    """
    获取日志目录
    :return:
    """
    return get_project_dir().joinpath("logs")


def get_data_dir() -> Path:
    """
    获取数据目录
    :return:
    """
    return get_project_dir().joinpath("data")


def get_scripts_dir() -> Path:
    """
    获取脚本目录
    :return:
    """
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir


def get_scripts_data(file: str) -> Path:
    """
    获取脚本数据
    :param file: 数据名词
    :return:
    """
    scripts_data_dir = get_scripts_dir().joinpath("data")
    return scripts_data_dir.joinpath(file)


def get_scripts_output(file: str) -> Path:
    """
    获取脚本输出
    :param file:
    :return:
    """
    scripts_output_dir = get_scripts_dir().joinpath("output")
    scripts_output_dir.mkdir(exist_ok=True, parents=True)
    return scripts_output_dir.joinpath(file)


if __name__ == '__main__':
    print(get_scripts_data("image.nii.gz"))
