数据处理：process\
特征工程：build_baseline_tensors，没有实际执行意义，被训练时或评价数据处理好不好时被调用\
它们的配置文件在configs下\
数据处理+特征工程的评估：l0和l1，分析结果在outputs下\
训练侧：training\
分为三种协议，每个里面包含超参搜索的yaml配置文件，和运行脚本train

消融实验启动：run_ablation，配置是full.yaml