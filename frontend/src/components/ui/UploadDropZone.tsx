import { useCallback, useState } from "react";

type UploadDropzoneProps = {
  onFileSelect: (file: File) => void;
};

export default function UploadDropzone({
  onFileSelect,
}: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files?.[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleFileChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
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
        isDragging
          ? "border-blue-500 bg-blue-50"
          : "border-gray-300"
      }`}
    >
      <div className="text-4xl mb-4">⬆️</div>

      <p className="font-medium mb-1">
        Choose a file to upload
      </p>
      <p className="text-sm text-gray-500 mb-4">
        Drag and drop your dataset or click to browse
      </p>

      <label className="inline-block">
        <span className="px-4 py-2 bg-blue-500 text-white rounded cursor-pointer">
          Browse Files
        </span>
        <input
          type="file"
          className="hidden"
          accept=".csv,.xes"
          onChange={handleFileChange}
        />
      </label>
    </div>
  );
}
