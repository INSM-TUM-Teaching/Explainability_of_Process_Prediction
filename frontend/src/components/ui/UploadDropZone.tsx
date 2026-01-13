import { useCallback, useState } from "react";

type UploadDropzoneProps = {
  onFileSelect: (file: File) => void;
  accept?: string;
  disabled?: boolean;
};

export default function UploadDropzone({
  onFileSelect,
  accept = ".csv,.xes",
  disabled = false,
}: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (disabled) return;
      setIsDragging(false);

      const file = e.dataTransfer.files?.[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [disabled, onFileSelect]
  );

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleFileChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (disabled) return;
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`border-2 border-dashed rounded-lg p-10 text-center transition ${
        disabled
          ? "border-gray-200 bg-gray-50 opacity-70"
          : isDragging
          ? "border-blue-500 bg-blue-50"
          : "border-gray-300"
      }`}
    >
      <div className="text-4xl mb-4">⬆️</div>

      <p className="font-medium mb-1">
        Choose a file to upload
      </p>
      <p className="text-sm text-gray-500 mb-4">
        {disabled
          ? "Select a file format first to enable upload."
          : "Drag and drop your dataset or click to browse"}
      </p>

      <label className="inline-block">
        <span
          className={`px-4 py-2 rounded cursor-pointer ${
            disabled ? "bg-gray-300 text-gray-700" : "bg-blue-500 text-white"
          }`}
        >
          Browse Files
        </span>
        <input
          type="file"
          className="hidden"
          accept={accept}
          onChange={handleFileChange}
          disabled={disabled}
        />
      </label>
    </div>
  );
}
