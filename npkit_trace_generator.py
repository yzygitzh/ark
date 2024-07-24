# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import json

from queue import Queue


NPKIT_EVENT_TIME_SYNC_GPU = 0x1
NPKIT_EVENT_TIME_SYNC_CPU = 0x2
NPKIT_EVENT_EXECUTOR_OP_BASE = 0x3


def parse_ark_tasks(ark_plan_path):
    with open(ark_plan_path, "r") as f:
        ark_plan = json.load(f)
        ark_tasks = {}
        for task_info in ark_plan["TaskInfos"]:
            ark_tasks[task_info["Id"]] = {
                "id": task_info["Id"],
                "num_warps": task_info["NumWarps"],
                "sram_bytes": task_info["SramBytes"],
                "op_names": ";".join([x["Name"] for x in task_info['Ops']]),
                "op_types": ";".join([x["Type"] for x in task_info['Ops']])
            }
        return ark_tasks


def parse_gpu_clock_scale(gpu_clock_file_path):
    with open(gpu_clock_file_path, "r") as f:
        freq_in_khz = f.read()
        return float(freq_in_khz) * 1e3 / 1e6


def parse_cpu_clock_scale(cpu_clock_den_file_path, cpu_clock_num_file_path):
    with open(cpu_clock_num_file_path, "r") as f:
        num = float(f.read())
    with open(cpu_clock_den_file_path, "r") as f:
        den = float(f.read())
    return den / num / 1e6


def parse_gpu_event(event_bytes):
    return {
        "id": int.from_bytes(event_bytes[0:4], byteorder="little", signed=False),
        "rsvd": int.from_bytes(event_bytes[4:8], byteorder="little", signed=False),
        "timestamp": int.from_bytes(event_bytes[8:16], byteorder="little", signed=False),
    }


def parse_gpu_event_file(npkit_dump_dir, ark_tasks, rank, buf_idx, gpu_clock_scale, cpu_clock_scale):
    gpu_event_file_path = os.path.join(npkit_dump_dir, "gpu_events_rank_%d_buf_%d" % (rank, buf_idx))
    raw_event_size = 16
    curr_cpu_base_time = None
    curr_gpu_base_time = None
    gpu_events = []
    event_type_to_seq = {}
    with open(gpu_event_file_path, "rb") as f:
        raw_content = f.read()
        raw_content_size = len(raw_content)
        raw_content_idx = 0
        while raw_content_idx < raw_content_size:
            parsed_gpu_event = parse_gpu_event(raw_content[raw_content_idx : raw_content_idx + raw_event_size])
            if parsed_gpu_event["id"] == NPKIT_EVENT_TIME_SYNC_CPU:
                curr_cpu_base_time = parsed_gpu_event["timestamp"] / cpu_clock_scale
                curr_gpu_base_time = None
            elif parsed_gpu_event["id"] == NPKIT_EVENT_TIME_SYNC_GPU:
                if curr_gpu_base_time is None:
                    curr_gpu_base_time = parsed_gpu_event["timestamp"] / gpu_clock_scale
            else:
                if curr_gpu_base_time is None:
                    curr_gpu_base_time = parsed_gpu_event["timestamp"] / gpu_clock_scale
                task_id = (parsed_gpu_event["id"] - NPKIT_EVENT_EXECUTOR_OP_BASE) // 2
                event_type = "task_%d" % task_id
                phase = "B" if (parsed_gpu_event["id"] % 2) else "E"
                gpu_events.append(
                    {
                        "ph": phase,
                        "ts": curr_cpu_base_time + parsed_gpu_event["timestamp"] / gpu_clock_scale - curr_gpu_base_time,
                        "pid": rank,
                        "tid": buf_idx + 1,
                    }
                )
                if phase == "B":
                    if event_type not in event_type_to_seq:
                        event_type_to_seq[event_type] = 0
                    gpu_events[-1].update(
                        {
                            "name": event_type,
                            "cat": "GPU",
                            "args": {
                                "rank": rank,
                                "buf_idx": buf_idx,
                                "seq": event_type_to_seq[event_type],
                                "rsvd_0": parsed_gpu_event["rsvd"],
                            },
                        }
                    )
                    gpu_events[-1]["args"].update(ark_tasks[task_id])
                    event_type_to_seq[event_type] += 1
                else:
                    gpu_events[-1]["args"] = {"rsvd": parsed_gpu_event["rsvd"]}
                    delta_time = gpu_events[-1]["ts"] - gpu_events[-2]["ts"]
            raw_content_idx += raw_event_size
    return gpu_events

def convert_npkit_dump_to_trace(npkit_dump_dir, ark_tasks, output_dir):
    files_in_dump_dir = next(os.walk(npkit_dump_dir))[2]
    gpu_event_files = [x for x in files_in_dump_dir if x.startswith("gpu_events_rank_")]
    cpu_event_files = [x for x in files_in_dump_dir if x.startswith("cpu_events_rank_")]

    ranks = list(set([int(x.split("_rank_")[1].split("_")[0]) for x in gpu_event_files]))
    buf_indices = list(set([int(x.split("_buf_")[1].split("_")[0]) for x in gpu_event_files]))
    channels = list(set([int(x.split("_channel_")[1].split("_")[0]) for x in cpu_event_files]))

    trace = {"traceEvents": []}

    for rank in ranks:
        cpu_clock_den_file_path = os.path.join(npkit_dump_dir, "cpu_clock_period_den_rank_%d" % rank)
        cpu_clock_num_file_path = os.path.join(npkit_dump_dir, "cpu_clock_period_num_rank_%d" % rank)
        cpu_clock_scale = parse_cpu_clock_scale(cpu_clock_den_file_path, cpu_clock_num_file_path)

        gpu_clock_file_path = os.path.join(npkit_dump_dir, "gpu_clock_rate_rank_%d" % rank)
        gpu_clock_scale = parse_gpu_clock_scale(gpu_clock_file_path)

        for buf_idx in buf_indices:
            gpu_events = parse_gpu_event_file(
                npkit_dump_dir, ark_tasks, rank, buf_idx, gpu_clock_scale, cpu_clock_scale
            )
            trace["traceEvents"].extend(gpu_events)

    trace["traceEvents"].sort(key=lambda x: x["ts"])
    trace["displayTimeUnit"] = "ns"

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "npkit_event_trace.json"), "w") as f:
        json.dump(trace, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npkit_dump_dir", type=str, required=True, help="NPKit dump directory.")
    parser.add_argument("--ark_plan_path", type=str, required=True, help="Path to ARK plan JSON.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    args = parser.parse_args()

    ark_tasks = parse_ark_tasks(args.ark_plan_path)
    convert_npkit_dump_to_trace(args.npkit_dump_dir, ark_tasks, args.output_dir)
