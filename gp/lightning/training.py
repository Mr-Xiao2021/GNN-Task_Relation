import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from gp.lightning.metric import EvalKit
from gp.utils.utils import dict_res_summary, load_pretrained_state

def lightning_fit(
    logger,                # WandbLogger — 日志记录器
    model,                 # GraphPredLightning — LightningModule
    data_module,           # DataModule — 提供 train/val/test 的 DataLoader
    metrics: EvalKit,      # 评估套件 — 包含 evlter、loss、eval_func 等
    num_epochs,            # 训练轮数，如 50
    profiler=None,         # 性能分析器，默认不使用
    cktp_prefix="",        # checkpoint 文件名前缀
    load_best=True,        # 训练结束后是否加载最优 checkpoint
    prog_freq=20,          # 每 N 步打印一次日志
    test_rep=1,            # 验证/测试重复次数（用于报告均值和标准差）
    save_model=True,       # 是否保存 checkpoint
    prog_bar=True,         # 是否显示进度条
    accelerator="auto",    # 加速器：GPU/CPU 自动检测
    detect_anomaly=False,  # 是否检测梯度异常（调试用，很慢）
    reload_freq=0,         # 每 N 个 epoch 重建 DataLoader（本项目覆盖为 1）
    val_interval=1,        # 每 N 个 epoch 做一次验证
    strategy=None,         # 分布式策略："deepspeed_stage_2" / "auto"
):
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    if save_model: # 保存最优模型回调
        callbacks.append(
            ModelCheckpoint(
                monitor=metrics.val_metric,
                mode=metrics.eval_mode,
                save_last=True, # 额外保存最后一个 epoch 的模型
                filename=cktp_prefix + "{epoch}-{step}",
            )
        )

    trainer = Trainer(
        accelerator=accelerator,                # "auto" → 有 GPU 用 GPU
        strategy=strategy,                      # "auto" 或 "deepspeed_stage_2"
        max_epochs=num_epochs,                  # 50 个 epoch
        callbacks=callbacks,                    # [TQDMProgressBar, ModelCheckpoint]
        logger=logger,                          # WandbLogger
        log_every_n_steps=prog_freq,            # 每 20 步 log 一次
        profiler=profiler,                      # None → 不分析
        enable_checkpointing=save_model,        # True → 启用 checkpoint
        enable_progress_bar=prog_bar,           # True → 显示进度条
        detect_anomaly=detect_anomaly,          # False → 不检测梯度异常
        reload_dataloaders_every_n_epochs=reload_freq,  # 1 → 每个 epoch 重建 DataLoader
        check_val_every_n_epoch=val_interval,            # 1 → 每个 epoch 验证
    )
    """
    每个 epoch：training_step × N 次 → on_train_epoch_end
    每 val_interval 个 epoch：validation_step × M 次 → on_validation_epoch_end
    每次验证：ModelCheckpoint 比较指标，保存最优
    学习率按 lr_scheduler 衰减
    """
    trainer.fit(model, datamodule=data_module)

    if load_best:
        model_dir = trainer.checkpoint_callback.best_model_path
        deep_speed = False
        if strategy[:9] == "deepspeed":
            deep_speed = True
        state_dict = load_pretrained_state(model_dir, deep_speed)
        model.load_state_dict(state_dict)


    val_col = []
    for i in range(test_rep):
        val_col.append(
            trainer.validate(model, datamodule=data_module, verbose=False)[0]
        )

    val_res = dict_res_summary(val_col)
    for met in val_res:
        val_mean = np.mean(val_res[met])
        val_std = np.std(val_res[met])
        print("{}:{:f}±{:f}".format(met, val_mean, val_std))

    target_val_mean = np.mean(val_res[metrics.val_metric])
    target_val_std = np.std(val_res[metrics.val_metric])

    test_col = []
    for i in range(test_rep):
        test_col.append(
            trainer.test(model, datamodule=data_module, verbose=False)[0]
        )

    test_res = dict_res_summary(test_col)
    for met in test_res:
        test_mean = np.mean(test_res[met])
        test_std = np.std(test_res[met])
        print("{}:{:f}±{:f}".format(met, test_mean, test_std))

    target_test_mean = np.mean(test_res[metrics.test_metric])
    target_test_std = np.std(test_res[metrics.test_metric])
    return [target_val_mean, target_val_std], [
        target_test_mean,
        target_test_std,
    ]


def lightning_test(
    logger,
    model,
    data_module,
    metrics: EvalKit,
    model_dir: str,
    strategy="auto",
    profiler=None,
    prog_freq=20,
    test_rep=1,
    prog_bar=True,
    accelerator="auto",
    detect_anomaly=False,
    deep_speed=True,
):
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    trainer = Trainer(
        accelerator=accelerator,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=prog_freq,
        profiler=profiler,
        enable_progress_bar=prog_bar,
        detect_anomaly=detect_anomaly,
    )
    state_dict = load_pretrained_state(model_dir, deep_speed)
    model.load_state_dict(state_dict)

    val_col = []
    for i in range(test_rep):
        val_col.append(
            trainer.validate(model, datamodule=data_module, verbose=False)[0]
        )

    val_res = dict_res_summary(val_col)
    for met in val_res:
        val_mean = np.mean(val_res[met])
        val_std = np.std(val_res[met])
        print("{}:{:f}±{:f}".format(met, val_mean, val_std))

    target_val_mean = np.mean(val_res[metrics.val_metric])
    target_val_std = np.std(val_res[metrics.val_metric])

    test_col = []
    for i in range(test_rep):
        test_col.append(
            trainer.test(model, datamodule=data_module, verbose=False)[0]
        )

    test_res = dict_res_summary(test_col)
    for met in test_res:
        test_mean = np.mean(test_res[met])
        test_std = np.std(test_res[met])
        print("{}:{:f}±{:f}".format(met, test_mean, test_std))

    target_test_mean = np.mean(test_res[metrics.test_metric])
    target_test_std = np.std(test_res[metrics.test_metric])
    return [target_val_mean, target_val_std], [
        target_test_mean,
        target_test_std,
    ]
