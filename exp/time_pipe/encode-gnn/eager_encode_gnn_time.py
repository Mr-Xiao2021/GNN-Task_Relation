"""Measure per-batch eager text encoding and GNN inference time."""

import time

import torch

import timing_utils as utils

from models.model import BinGraphAttModel, BinGraphModel


def base_forward(params):
    return BinGraphAttModel.forward if params.JK == "none" else BinGraphModel.forward


def eager_forward(model, g, gnn_forward):
    g = model._encode_graph_texts(g)
    return gnn_forward(model, g)


def run_warmup(loader, model, params, device, warmup_batches):
    if warmup_batches == 0:
        return
    gnn_forward = base_forward(params)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= warmup_batches:
                break
            eager_forward(model, utils.move_batch(batch, device), gnn_forward)
    utils.synchronize(device)


def time_eager_batches(loader, model, params, metric, device, batch_num):
    # Tokenization and text encoder wall time over the measured batches.
    encode_seconds = 0.0
    text_prepare_seconds = 0.0
    text_restore_seconds = 0.0
    # GNN forward wall time over the measured batches.
    gnn_seconds = 0.0
    # Number of batches and PyG graphs that actually enter the timing result.
    measured_batches = 0
    measured_graphs = 0
    # Raw node/edge text counts before per-batch text deduplication.
    encoded_node_texts = 0
    encoded_edge_texts = 0
    unique_encoded_texts = 0
    text_encoder_batches = 0
    # Number of scalar values in the model output.
    output_values = 0
    # Call the non-LLM parent forward after eager text encoding is complete.
    gnn_forward = base_forward(params)

    with torch.inference_mode():
        for batch in loader:
            if batch_num != -1 and measured_batches >= batch_num:
                break
            batch = utils.move_batch(batch, device)
            encoded_node_texts += len(batch.x)
            encoded_edge_texts += len(batch.edge_attr)

            utils.synchronize(device)
            start = time.perf_counter()
            unique_texts, text_mapping, num_nodes = model._prepare_graph_texts(batch)
            text_prepare_seconds += time.perf_counter() - start
            unique_encoded_texts += len(unique_texts)
            text_batch_size = model.text_batch_size
            if text_batch_size <= 0:
                text_batch_size = len(unique_texts)
            text_encoder_batches += (len(unique_texts) + text_batch_size - 1) // text_batch_size

            start = time.perf_counter()
            text_features = model._encode_texts(unique_texts)
            utils.synchronize(device)
            encode_seconds += time.perf_counter() - start

            start = time.perf_counter()
            batch = model._restore_graph_text_features(
                batch, text_features, text_mapping, num_nodes
            )
            utils.synchronize(device)
            text_restore_seconds += time.perf_counter() - start

            start = time.perf_counter()
            output = gnn_forward(model, batch)
            utils.synchronize(device)
            gnn_seconds += time.perf_counter() - start

            measured_batches += 1
            measured_graphs += batch.num_graphs
            output_values += output.numel()
            # Use the selected dataset's run_cdm evaluator outside the timed region.
            utils.update_eval_metric(metric, output, batch)

    if measured_batches == 0:
        raise RuntimeError("No batches were measured; check split, batch_size, and drop_last")
    return {
        "encode_seconds": encode_seconds,
        "text_prepare_seconds": text_prepare_seconds,
        "text_restore_seconds": text_restore_seconds,
        "gnn_seconds": gnn_seconds,
        "measured_batches": measured_batches,
        "measured_graphs": measured_graphs,
        "encoded_node_texts": encoded_node_texts,
        "encoded_edge_texts": encoded_edge_texts,
        "unique_encoded_texts": unique_encoded_texts,
        "text_encoder_batches": text_encoder_batches,
        "output_values": output_values,
    }


def main():
    args = utils.parse_args("Time eager per-batch encoding and GNN inference")
    params = utils.load_params(args, load_texts=True)
    device = utils.resolve_device(args.device)

    data_start = time.perf_counter()
    _, data_module = utils.build_task_data(params, encoder=None)
    data_preparation_seconds = time.perf_counter() - data_start

    model_load_start = time.perf_counter()
    model = utils.build_eager_model(params)
    checkpoint_loaded = utils.load_model_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    utils.synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_start

    loader = utils.select_loader(data_module, args.split, args.loader_index)
    metric = utils.build_eval_metric(
        data_module, args.split, args.loader_index, device
    )
    run_warmup(loader, model, params, device, args.warmup_batches)
    timing = time_eager_batches(loader, model, params, metric, device, params.batch_num)
    metric_report = utils.eval_metric_report(metric)

    utils.make_report(
        "eager",
        params,
        args,
        device,
        checkpoint_loaded,
        timing["encode_seconds"],
        timing["gnn_seconds"],
        timing["measured_batches"],
        timing["measured_graphs"],
        timing["output_values"],
        extra={
            "encode_scope": "tokenization + text encoder forward on per-batch unique texts",
            "gnn_scope": "GNN forward on measured batches",
            "ratio_scope": "tokenization + text encoder versus GNN; text preparation/restore excluded",
            "text_prepare_seconds": timing["text_prepare_seconds"],
            "text_restore_seconds": timing["text_restore_seconds"],
            "eager_pipeline_seconds": (
                timing["text_prepare_seconds"]
                + timing["encode_seconds"]
                + timing["text_restore_seconds"]
                + timing["gnn_seconds"]
            ),
            "encoded_node_texts_before_dedup": timing["encoded_node_texts"],
            "encoded_edge_texts_before_dedup": timing["encoded_edge_texts"],
            "unique_encoded_texts": timing["unique_encoded_texts"],
            "text_encoder_batches": timing["text_encoder_batches"],
            "encode_seconds_per_batch": timing["encode_seconds"] / timing["measured_batches"],
            **metric_report,
            "metric_scope": "selected measured batches; excluded from timing",
            "encoder_model_load_seconds_excluded": model_load_seconds,
            "data_preparation_seconds_excluded": data_preparation_seconds,
        },
    )


if __name__ == "__main__":
    main()
