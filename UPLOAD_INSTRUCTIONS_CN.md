# 覆盖上传说明

本压缩包只包含需要替换或新增的代码文件，不包含原始圆柱 `.mat` 数据、NASA `.dat` 数据和模型权重。

1. 将 `cylinder_wake/` 中的同名文件覆盖到 GitHub 的 `cylinder_wake/`。
2. 新增 `train_dropout_ablation.py` 和 `wake_probe_spectrum.py`。
3. 将 `nasa_hump/hump_train.py`、`nasa_hump/hump_validation.py` 覆盖原文件。
4. 用新的 `README.md` 覆盖仓库根目录 README。
5. 将 `verify_manuscript_config.py` 上传到仓库根目录。
6. NASA 数据文件继续保留在 `nasa_hump/` 中，不要删除。
7. 上传代码后先运行 `python verify_manuscript_config.py`。
8. 随后必须重新训练五个 dropout 模型并重新生成消融 CSV；不能仅上传代码就继续使用旧 Table 5 数字。

重要：本包没有伪造或预生成任何训练结果。实际运行结果若与论文表格数字不一致，应以新运行结果更新论文和 Response Letter。
