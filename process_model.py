import os
from datetime import datetime
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer

def load_event_log(xes_path: str):
    return xes_importer.apply(xes_path)

def ensure_output_folder(folder_path: str):
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def petri_from_log(event_log):
    tree = inductive_miner.apply(event_log, variant=inductive_miner.Variants.IMf)
    net, im, fm = pt_converter.apply(tree, variant=pt_converter.Variants.TO_PETRI_NET)
    return net, im, fm

def bpmn_from_log(event_log):
    tree = inductive_miner.apply(event_log, variant=inductive_miner.Variants.IMf)
    bpmn_graph = pt_converter.apply(tree, variant=pt_converter.Variants.TO_BPMN)
    return bpmn_graph

def save_petri_jpg(net, im, fm, output_folder: str, base_name: str = "bpi2017_petri"):
    ensure_output_folder(output_folder)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_folder, f"{base_name}_{ts}.jpg")
    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.save(gviz, path)
    return path

def save_bpmn_jpg(bpmn_graph, output_folder: str, base_name: str = "bpi2017_bpmn"):
    ensure_output_folder(output_folder)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_folder, f"{base_name}_{ts}.jpg")
    gviz = bpmn_visualizer.apply(bpmn_graph)
    bpmn_visualizer.save(gviz, path)
    return path

def build_and_save_petri(xes_path: str, output_folder: str):
    log = load_event_log(xes_path)
    net, im, fm = petri_from_log(log)
    return save_petri_jpg(net, im, fm, output_folder)

def build_and_save_bpmn(xes_path: str, output_folder: str):
    log = load_event_log(xes_path)
    bpmn_graph = bpmn_from_log(log)
    return save_bpmn_jpg(bpmn_graph, output_folder)

def build_and_save_all(xes_path: str, output_folder: str):
    petri_path = build_and_save_petri(xes_path, output_folder)
    bpmn_path = build_and_save_bpmn(xes_path, output_folder)
    return {"petri_jpg": petri_path, "bpmn_jpg": bpmn_path}


