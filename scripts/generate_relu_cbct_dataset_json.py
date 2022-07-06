import json
from pathlib import Path

if __name__ == '__main__':
    dataset_dir = Path("/media/3TB/data/xiaoliutech/relu_cbct_dataset")
    # 匹配待处理的数据和标签
    all_images = list(dataset_dir.joinpath("images").rglob("*.dcm"))
    all_label_dirs = []
    for label_dir in dataset_dir.joinpath("labels").iterdir():
        for item in label_dir.iterdir():
            if item.is_dir():
                all_label_dirs.append(item)
    images, label_dirs = [], []
    for image in all_images:
        if image.name in ["631533_no_cbct.dcm", "85073_p_cbct.dcm", "629068_In_cbct.dcm"]:
            # TODO 异常的数据暂时跳过，标签有问题，会导致nnDetection在预处理的时候卡死，需要数据清洗
            continue
        for label_dir in all_label_dirs:
            image_label_name = image.name.replace("_cbct.dcm", "_structures")
            if image_label_name == label_dir.name:
                images.append(image)
                label_dirs.append(label_dir)
                break
    write_data = [{
        "image": str(image),
        "label_dir": str(label_dir)
    } for image, label_dir in zip(images, label_dirs)]
    with open(dataset_dir.joinpath("dataset.json"), "w") as f:
        f.write(json.dumps(write_data, indent=4))
    print(f"生成完毕，共{len(write_data)}个数据")
