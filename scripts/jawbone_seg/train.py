from monai.apps.auto3dseg import AutoRunner

if __name__ == '__main__':
    runner = AutoRunner(input={"name": "Task500_Jawbone",
                                "task": "segmentation",
                                "modality": "CT",
                                "datalist": "/home/yujiannan/Projects/MONAI/scripts/jawbone_seg/dataset.json",
                                "dataroot": "/media/DATA2/yujiannan/20221124/",
                                "class_names": ["upper", "lower"]},
                        work_dir="/home/yujiannan/Projects/MONAI/work_dir",
                        templates_path_or_url="/home/yujiannan/Projects/MONAI/work_dir/",
                        analyze=False,
                        algo_gen=False)
    runner.run()