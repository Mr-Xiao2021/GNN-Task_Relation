"""Measure global offline text encoding and batched GNN inference time."""

import gc
import time

import torch

import timing_utils as utils


SentenceEncoder = utils.project_utils.SentenceEncoder


def run_warmup(loader, model, device, warmup_batches):
    if warmup_batches == 0:
        return
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= warmup_batches:
                break
            model(utils.move_batch(batch, device))
    utils.synchronize(device)


def time_model_batches(loader, model, metric, device, batch_num):
    elapsed = 0.0
    measured_batches = 0
    measured_graphs = 0
    output_values = 0
    with torch.inference_mode():
        for batch in loader:
            if batch_num != -1 and measured_batches >= batch_num:
                break
            batch = utils.move_batch(batch, device)
            utils.synchronize(device)
            start = time.perf_counter()
            output = model(batch)
            utils.synchronize(device)
            elapsed += time.perf_counter() - start
            measured_batches += 1
            measured_graphs += batch.num_graphs
            output_values += output.numel()
            # Use the selected dataset's run_cdm evaluator outside the timed region.
            utils.update_eval_metric(metric, output, batch)
    if measured_batches == 0:
        raise RuntimeError("No batches were measured; check split, batch_size, and drop_last")
    return (
        elapsed,
        measured_batches,
        measured_graphs,
        output_values,
    )


def load_cached_texts(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def count_texts(value):
    if isinstance(value, str):
        return 1
    return sum(count_texts(item) for item in value)


def replay_global_offline_encode(tasks, encoder):
    # Exclude texts.pkl disk I/O, matching eager timing which starts after batch loading.
    encode_jobs = [
        (dataset, load_cached_texts(dataset.processed_paths[1]))
        for dataset in tasks.dataset.values()
    ]
    encoded_texts = sum(count_texts(texts) for _, texts in encode_jobs)

    utils.synchronize(encoder.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for dataset, texts in encode_jobs:
            embeddings = dataset.text2feature(texts)
            del embeddings
    utils.synchronize(encoder.device)
    encode_seconds = time.perf_counter() - start
    return encode_seconds, encoded_texts


def main():
    args = utils.parse_args("Time offline global encoding and GNN inference")
    params = utils.load_params(args, load_texts=False)
    device = utils.resolve_device(args.device)

    encoder_load_start = time.perf_counter()
    encoder = SentenceEncoder(
        params.llm_name,
        batch_size=params.llm_b_size,
        max_length=params.llm_max_length,
    )
    if encoder.device != device:
        encoder.device = device
        encoder.model.to(device)
    utils.synchronize(device)
    encoder_load_seconds = time.perf_counter() - encoder_load_start

    data_start = time.perf_counter()
    tasks, data_module = utils.build_task_data(params, encoder)
    data_preparation_seconds = time.perf_counter() - data_start

    encode_seconds, encoded_texts = replay_global_offline_encode(tasks, encoder)
    encoder.flush_model()
    gc.collect()

    model = utils.build_offline_model(params)
    checkpoint_loaded = utils.load_model_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    loader = utils.select_loader(data_module, args.split, args.loader_index)
    metric = utils.build_eval_metric(
        data_module, args.split, args.loader_index, device
    )
    run_warmup(loader, model, device, args.warmup_batches)
    
    (
        gnn_seconds,
        measured_batches,
        measured_graphs,
        output_values,
    ) = time_model_batches(loader, model, metric, device, params.batch_num)
    metric_report = utils.eval_metric_report(metric)

    utils.make_report(
        "offline",
        params,
        args,
        device,
        checkpoint_loaded,
        encode_seconds,
        gnn_seconds,
        measured_batches,
        measured_graphs,
        output_values,
        extra={
            "encode_scope": "all unique base-dataset texts (replayed from texts.pkl)",
            "gnn_scope": "GNN forward on measured batches",
            "ratio_scope": "global one-time encode versus selected-batch GNN inference",
            "encoded_texts": encoded_texts,
            **metric_report,
            "metric_scope": "selected measured batches; excluded from timing",
            "encoder_model_load_seconds_excluded": encoder_load_seconds,
            "data_preparation_seconds_excluded": data_preparation_seconds,
        },
    )


if __name__ == "__main__":
    main()
