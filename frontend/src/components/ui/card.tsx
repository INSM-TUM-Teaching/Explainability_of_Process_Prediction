import { ReactNode } from "react";

type CardProps = {
  title?: string;
  description?: string;
  children: ReactNode;
};

export default function Card({
  title,
  description,
  children,
}: CardProps) {
  return (
    <div className="bg-white border rounded-lg p-6 shadow-sm">
      {title && (
        <h3 className="text-lg font-semibold mb-1">
          {title}
        </h3>
      )}

      {description && (
        <p className="text-sm text-gray-500 mb-4">
          {description}
        </p>
      )}

      {children}
    </div>
  );
}
