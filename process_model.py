import os
from datetime import datetime
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def load_event_log(xes_path: str):
    return xes_importer.apply(xes_path)


def ensure_output_folder(folder_path: str):
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def generate_dfg(event_log):
    dfg_freq = dfg_discovery.apply(event_log, variant=dfg_discovery.Variants.FREQUENCY)
    dfg_perf = dfg_discovery.apply(event_log, variant=dfg_discovery.Variants.PERFORMANCE)
    return dfg_freq, dfg_perf


def save_dfg_visual(dfg, log, output_folder: str, variant: str = "frequency"):
    ensure_output_folder(output_folder)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_folder, f"bpi2017_dfg_{variant}_{ts}.jpg")

    if variant == "frequency":
        gviz = dfg_visualizer.apply(dfg, log=log, variant=dfg_visualizer.Variants.FREQUENCY)
    else:
        gviz = dfg_visualizer.apply(dfg, log=log, variant=dfg_visualizer.Variants.PERFORMANCE)

    dfg_visualizer.save(gviz, path)
    return path


def petri_from_log(event_log):
    tree = inductive_miner.apply(event_log, variant=inductive_miner.Variants.IMf)
    net, im, fm = pt_converter.apply(tree, variant=pt_converter.Variants.TO_PETRI_NET)
    return net, im, fm


def save_petri_jpg(net, im, fm, output_folder: str, base_name: str = "bpi2017_petri"):
    ensure_output_folder(output_folder)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_folder, f"{base_name}_{ts}.jpg")
    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.save(gviz, path)
    return path


def build_and_save_all(xes_path: str, output_folder: str):
    log = load_event_log(xes_path)
    dfg_freq, dfg_perf = generate_dfg(log)
    net, im, fm = petri_from_log(log)

    freq_path = save_dfg_visual(dfg_freq, log, output_folder, "frequency")
    perf_path = save_dfg_visual(dfg_perf, log, output_folder, "performance")
    petri_path = save_petri_jpg(net, im, fm, output_folder)

    return {
        "frequency_jpg": freq_path,
        "performance_jpg": perf_path,
        "petri_jpg": petri_path
    }


def build_and_save_dfg_visuals(xes_path: str, output_folder: str):
    return build_and_save_all(xes_path, output_folder)


def build_and_save_petri(xes_path: str, output_folder: str):
    return build_and_save_all(xes_path, output_folder)

