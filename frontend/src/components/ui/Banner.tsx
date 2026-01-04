type BannerProps = {
  variant: "info" | "success";
  title: string;
  description: string;
};

export default function Banner({
  variant,
  title,
  description,
}: BannerProps) {
  const styles =
    variant === "success"
      ? "bg-green-100 border-green-300 text-green-800"
      : "bg-purple-100 border-purple-300 text-purple-800";

  return (
    <div className={`border rounded-lg p-4 ${styles}`}>
      <div className="font-semibold">{title}</div>
      <div className="text-sm">{description}</div>
    </div>
  );
}
