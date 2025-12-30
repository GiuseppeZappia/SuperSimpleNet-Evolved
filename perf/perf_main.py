import csv
import gc
import sys
import timeit

from tqdm import tqdm

sys.path.append("../")

from model.supersimplenet import SuperSimpleNet
import torch


def params(backbone_name, case):
    config = {
        "backbone": backbone_name, # added in order to eavluate for a specific backbone
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise_std": 0.015,
        "stop_grad": False,
        "non_linear_adaptor": (case == "B") # Flag for case B
    }
    model = SuperSimpleNet(image_size=(256, 256), config=config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Using: {backbone_name}| Case: {case}")
    print("Total params:", total_params, "Trainable params:", trainable_params)


def prepare_image(batched=False):
    if batched:
        img = torch.randn(16, 3, 256, 256, dtype=torch.float16)
    else:
        img = torch.randn(1, 3, 256, 256, dtype=torch.float16)

    return img


def prepare_model(backbone_name, case):
    config = {
        "backbone": backbone_name, # added in order to evaluate for a specific backbone
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise_std": 0.015,
        "stop_grad": False,
        "non_linear_adaptor": (case == "B") # Flag for case B
    }
    model = SuperSimpleNet(image_size=(256, 256), config=config)
    # model.load_model("./pcb1/weights.pt")
    model.to("cuda")
    model.to(torch.float16)
    model.eval()

    return model


@torch.no_grad()
def inference_speed(backbone, case, reps=1000):
    model = prepare_model(backbone, case)
    img = prepare_image()

    # first - warmup
    for i in tqdm(range(reps), desc="Warmup"):
        img = img.to("cpu")

        img = img.to("cuda")
        out = model(img)
        out = out[0].to("cpu"), out[1].to("cpu")

    total_time = 0
    # next - real
    for i in tqdm(range(reps), desc="Timing inference"):
        img = img.to("cpu")

        t0 = timeit.default_timer()

        img = img.to("cuda")
        out = model(img)
        out = out[0].to("cpu"), out[1].to("cpu")

        t1 = timeit.default_timer()
        total_time += t1 - t0

    # * 1000 to get ms
    ms = total_time * 1000 / reps
    print("Speed in ms:", ms)
    return ms


@torch.no_grad()
def throughput(backbone, case, reps=1000):
    model = prepare_model(backbone, case)
    img = prepare_image(batched=True)

    # first - warmup
    for i in tqdm(range(reps), desc="Warmup"):
        img = img.to("cpu")

        img = img.to("cuda")
        out = model(img)
        out = out[0].to("cpu"), out[1].to("cpu")

    total_time = 0
    # next - real
    for i in tqdm(range(reps), desc="Throughput"):
        img = img.to("cpu")

        t0 = timeit.default_timer()

        img = img.to("cuda")
        out = model(img)
        out = out[0].to("cpu"), out[1].to("cpu")

        t1 = timeit.default_timer()
        total_time += t1 - t0

    thru = 16 * reps / total_time
    print("Throughput:", thru)
    return thru


@torch.no_grad()
def memory(backbone, case, reps=1000):
    model = prepare_model(backbone, case)
    img = prepare_image()
    img = img.to("cuda")

    # first - warmup
    for i in tqdm(range(reps), desc="Warmup"):
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        out = model(img)

        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_memory = 0
    # next - real
    for i in tqdm(range(reps), desc="Memory calc"):
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        out = model(img)

        total_memory += torch.cuda.max_memory_reserved()

        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # MB -> 10**6 bytes, then "reps" runs
    mbs = total_memory / (10**6) / reps
    print("Memory in MB:", mbs)
    return mbs


@torch.no_grad()
def flops(backbone, case, reps=1000):
    model = prepare_model(backbone, case)
    img = prepare_image()
    img = img.to("cuda")

    # first - warmup
    out = model(img)

    # real - don't need reps as the result is always same
    with torch.profiler.profile(with_flops=True) as prof:
        out = model(img)
    tflops = sum(x.flops for x in prof.key_averages()) / 1e9
    print("TFLOPS:", tflops)

    return tflops


# new main function that takes backbone from CLI
def main():
    if len(sys.argv) < 4:
            raise ValueError("Usage: python perf_main.py <gpu_model> <backbone> <case>")

    gpu_model = sys.argv[1]
    backbone = sys.argv[2]
    case = sys.argv[3] # A o B

    cycles = 6
    reps = 1000
    torch.backends.cudnn.deterministic = True

    # Showing chosen backbone parameters before starting
    print("Backbone parameters and case:")
    params(backbone, case)

    # Modified name to distinguish results
    with open(f"perf_{gpu_model}_{backbone}_{case}.csv", "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["time", "throughput", "memory", "tflops"])
        for cyc in range(cycles):
            ms = inference_speed(backbone, case, reps)
            thru = throughput(backbone, case, reps)
            mbs = memory(backbone, case, reps)
            tflops = flops(backbone, case, reps)

            if cyc == 0: continue

            writer.writerow([ms, thru, mbs, tflops])
            print(f"--- Cycle {cyc} | {backbone} {case} ---")
            print(f"Speed: {ms:.2f} ms | Throughput: {thru:.2f} | Memory: {mbs:.2f} MB")

if __name__ == "__main__":
    main()
    # params()
