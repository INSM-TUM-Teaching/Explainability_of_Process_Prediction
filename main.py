from process_model import build_and_save_all

def main():
    xes_path = "BPI_Models/BPI_logs_xes/BPI_2020_Log_RequestForPayment.xes"
    output_folder = "BPI_Models/process_models/BPI_2020_Log_RequestForPayment"

    results = build_and_save_all(xes_path, output_folder)

    print("Frequency DFG saved at:", results["frequency_jpg"])
    print("Performance DFG saved at:", results["performance_jpg"])
    print("Petrinet saved at:", results["petri_jpg"])

if __name__ == "__main__":
    main()
