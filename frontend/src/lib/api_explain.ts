import { API_BASE } from "./api";

export async function explainOnDemand(runId: string, caseId: string, caseIndex: number, method: string) {
  const res = await fetch(`${API_BASE}/runs/${runId}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_id: caseId, case_index: caseIndex, method })
  });
  if (!res.ok) {
    const data = await res.json();
    throw new Error(data.detail || "Explain request failed");
  }
  return res.json();
}
