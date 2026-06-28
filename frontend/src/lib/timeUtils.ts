export function formatRemainingTime(days: number | string | null | undefined): string {
  if (days === null || days === undefined) return "N/A";
  
  const numericDays = typeof days === "string" ? parseFloat(days) : days;
  if (isNaN(numericDays)) return "N/A";
  
  const isNegative = numericDays < 0;
  const absDays = Math.abs(numericDays);
  
  if (absDays === 0) return "0 Days";

  const totalHours = absDays * 24;
  const d = Math.floor(totalHours / 24);
  const h = Math.floor(totalHours % 24);
  const m = Math.round((totalHours * 60) % 60);

  const parts = [];
  if (d > 0) parts.push(`${d} Day${d !== 1 ? 's' : ''}`);
  if (h > 0) parts.push(`${h} Hour${h !== 1 ? 's' : ''}`);
  
  if (d === 0 && h === 0 && m > 0) {
    parts.push(`${m} Minute${m !== 1 ? 's' : ''}`);
  }

  const result = parts.join(", ") || "0 Days";
  return isNegative ? `-${result}` : result;
}
