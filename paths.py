PATHS = {
    "av2": {
        "train": {
            "infos": "/data/ashen3/datasets/ArgoVerse2/Sensor/av2_mmdet3d_trainval/av2_infos_train.pkl",
            "dataset_dir": "/data/ashen3/datasets/ArgoVerse2/Sensor/train",
        },
        "val": {
            "prediction": "/home/ashen3/mmdet3d-lt3d/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2/predictions.pkl",
            "infos": "/data/ashen3/datasets/ArgoVerse2/Sensor/av2_mmdet3d_trainval/av2_infos_val.pkl",
            "dataset_dir": "/data/ashen3/datasets/ArgoVerse2/Sensor/val",
        },
        "test": {
            "prediction": "/home/ashen3/mmdet3d-lt3d/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2/test_predictions.pkl",
            "infos": "/data/ashen3/datasets/ArgoVerse2/Sensor/av2_mmdet3d_trainval/av2_infos_test.pkl",
            "dataset_dir": "/data/ashen3/datasets/ArgoVerse2/Sensor/test",
        },
    },
    "nuscenes": {
        "train": {
            "infos": "/data/ashen3/datasets/nuScenes/nusc_mmdet3d_trainval/nuscenes_infos_train.pkl",
            "dataset_dir": "/data/ashen3/datasets/nuScenes",
        },
        "val": {
            "prediction": "/home/ashen3/mmdet3d-lt3d/work_dirs/hv_pointpillars_fpn_sbn-all_4x8_2x_hierarchy_nus/predictions.pkl",
            "infos": "/data/ashen3/datasets/nuScenes/nusc_mmdet3d_trainval/nuscenes_infos_val.pkl",
            "dataset_dir": "/data/ashen3/datasets/nuScenes",
        },
    },
}
