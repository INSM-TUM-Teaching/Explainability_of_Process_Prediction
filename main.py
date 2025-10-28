from process_model import build_and_save_all
def main():
    xes_path = "BPI_Challenge_2017/BPI_Challenge_2017.xes"
    output_folder = "BPI_Models/process_models"
    results = build_and_save_all(xes_path, output_folder)
    print("Petri Net saved at:", results["petri_jpg"])
    print("BPMN Model saved at:", results["bpmn_jpg"])

if __name__ == "__main__":
    main()