// frontend/src/components/steps/Step1Upload.tsx
import { useState } from "react";
import Card from "../ui/card";
import UploadDropzone from "../ui/UploadDropZone";
import type { DatasetUploadResponse } from "../../lib/api";
import { uploadDataset } from "../../lib/api";

type Step1UploadProps = {
  uploadedFile: File | null;
  dataset: DatasetUploadResponse | null;
  onUploaded: (file: File, dataset: DatasetUploadResponse) => void;
  onClear?: () => void;
};

const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function getExt(name: string): string {
  const i = name.lastIndexOf(".");
  if (i === -1) return "";
  return name.slice(i + 1).toLowerCase();
}

export default function Step1Upload({
  uploadedFile,
  dataset,
  onUploaded,
  onClear,
}: Step1UploadProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    setError(null);

    // 1) size check
    if (file.size > MAX_UPLOAD_BYTES) {
      setError("File too large. Max allowed size is 100 MB.");
      return;
    }

    // 2) extension check
    const ext = getExt(file.name);
    if (ext !== "csv" && ext !== "xes") {
      setError("Unsupported format. Please upload a CSV or XES file.");
      return;
    }

    // NOTE: backend currently supports CSV only unless you add XES parsing server-side.
    if (ext === "xes") {
      setError("XES upload is not supported by the backend yet. Please upload CSV for now.");
      return;
    }

    setIsUploading(true);
    try {
      const resp = await uploadDataset(file);
      onUploaded(file, resp);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Upload failed.";
      setError(msg);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-8 w-full">
      <div>
        <h2 className="text-2xl font-semibold">Upload Dataset</h2>
        <p className="text-sm text-gray-500">
          Upload your event log dataset. We will validate size, format, and required columns.
        </p>
      </div>

      <Card>
        <div className="w-full">
          {!uploadedFile && (
            <div className="space-y-3">
              <UploadDropzone onFileSelect={handleUpload} />

              {isUploading && (
                <div className="text-sm text-gray-600">Uploading and parsing dataset…</div>
              )}

              {error && (
                <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                  <div className="font-medium">Upload blocked / failed</div>
                  <div className="mt-1 break-words">{error}</div>
                </div>
              )}
            </div>
          )}

          {uploadedFile && dataset && (
            <div className="w-full border border-green-300 bg-green-50 rounded-lg p-6 space-y-3">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="font-medium text-green-800">
                    Dataset Uploaded Successfully
                  </div>
                  <div className="text-sm text-green-700 mt-1">
                    {uploadedFile.name} ({formatBytes(uploadedFile.size)}) —{" "}
                    {dataset.num_events.toLocaleString()} events,{" "}
                    {dataset.num_cases.toLocaleString()} cases
                  </div>
                  <div className="text-xs text-green-700 mt-2">
                    <span className="font-medium">dataset_id:</span>{" "}
                    <span className="font-mono">{dataset.dataset_id}</span>
                  </div>
                </div>

                {onClear && (
                  <button
                    type="button"
                    onClick={onClear}
                    className="text-sm px-3 py-2 rounded-md border border-green-300 bg-white hover:bg-green-100"
                  >
                    Upload another file
                  </button>
                )}
              </div>

              <div className="text-sm text-gray-700 bg-white border border-green-200 rounded-md p-4">
                <div className="font-medium text-gray-800">Detected mapping</div>
                <div className="mt-2 text-xs text-gray-700">
                  Backend standardizes to: <span className="font-mono">CaseID</span>,{" "}
                  <span className="font-mono">Activity</span>,{" "}
                  <span className="font-mono">Timestamp</span>{" "}
                  (and optionally <span className="font-mono">Resource</span>).
                </div>
                <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
                  {Object.keys(dataset.detected_mapping).length === 0 ? (
                    <div className="text-gray-600">No renaming was needed.</div>
                  ) : (
                    Object.entries(dataset.detected_mapping).map(([from, to]) => (
                      <div
                        key={`${from}->${to}`}
                        className="flex items-center justify-between bg-gray-50 border border-gray-200 rounded px-3 py-2"
                      >
                        <span className="font-mono text-gray-700">{from}</span>
                        <span className="text-gray-500 mx-2">→</span>
                        <span className="font-mono text-gray-900">{to}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>

      <Card title="Dataset Requirements">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <ul className="text-sm text-gray-700 list-disc ml-5 space-y-1">
            <li>File size: Maximum 100 MB (checked in frontend + backend)</li>
            <li>Formats: CSV or XES (XES requires backend support; currently CSV only)</li>
            <li>Required columns: Case ID, Activity, Timestamp (validated in backend via auto-detection)</li>
          </ul>
        </div>
      </Card>
    </div>
  );
}
