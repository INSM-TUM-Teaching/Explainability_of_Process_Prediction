import { useEffect, useState } from "react";

import Card from "../ui/card";
import { artifactUrl } from "../../lib/api";

type PatternRow = {
  rank: number;
  match_count: number;
  accuracy: number;
  pattern: string;
  activity_count?: number;
};

type BestPatternsPanelProps = {
  runId: string;
  artifactPaths: string[];
};

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out;
}

function aggregateFromMatchedCsv(text: string): PatternRow[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const header = parseCsvLine(lines[0]);
  const patIdx = header.indexOf("matched_pattern");
  const correctIdx = header.indexOf("correct");
  if (patIdx < 0) return [];

  const stats = new Map<string, { count: number; hits: number }>();
  for (let i = 1; i < lines.length; i += 1) {
    const cols = parseCsvLine(lines[i]);
    const pat = cols[patIdx]?.trim();
    if (!pat) continue;
    const prev = stats.get(pat) ?? { count: 0, hits: 0 };
    prev.count += 1;
    if (correctIdx >= 0) {
      const c = cols[correctIdx]?.trim();
      if (c === "1" || c === "True" || c === "true") prev.hits += 1;
    }
    stats.set(pat, prev);
  }

  return Array.from(stats.entries())
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 15)
    .map(([pattern, s], idx) => ({
      rank: idx + 1,
      match_count: s.count,
      accuracy: s.count > 0 ? s.hits / s.count : 0,
      pattern,
      activity_count: pattern.split("→").filter((x) => x.trim()).length,
    }));
}

export default function BestPatternsPanel({ runId, artifactPaths }: BestPatternsPanelProps) {
  const [rows, setRows] = useState<PatternRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [source, setSource] = useState<string | null>(null);

  const summaryJson = artifactPaths.find((p) =>
    p.replace(/\\/g, "/").endsWith("explainability/top_patterns_summary.json")
  );
  const summaryCsv = artifactPaths.find((p) =>
    p.replace(/\\/g, "/").endsWith("explainability/top_patterns_summary.csv")
  );
  const matchedCsv = artifactPaths.find((p) =>
    p.replace(/\\/g, "/").endsWith("explainability/matched_patterns.csv")
  );

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setRows([]);
      setSource(null);

      try {
        if (summaryJson) {
          const res = await fetch(artifactUrl(runId, summaryJson));
          if (!res.ok) throw new Error(String(res.status));
          const data = (await res.json()) as PatternRow[];
          if (!cancelled && Array.isArray(data) && data.length > 0) {
            setRows(data.slice(0, 15));
            setSource("top_patterns_summary.json");
            return;
          }
        }

        const csvPath = summaryCsv ?? matchedCsv;
        if (csvPath) {
          const res = await fetch(artifactUrl(runId, csvPath));
          if (!res.ok) throw new Error(String(res.status));
          const text = await res.text();
          const aggregated =
            csvPath.includes("top_patterns_summary.csv")
              ? (() => {
                  const lines = text.trim().split(/\r?\n/);
                  if (lines.length < 2) return [];
                  const header = parseCsvLine(lines[0]);
                  const rankIdx = header.indexOf("rank");
                  const countIdx = header.indexOf("match_count");
                  const accIdx = header.indexOf("accuracy");
                  const patIdx = header.indexOf("pattern");
                  return lines.slice(1, 16).map((line, i) => {
                    const cols = parseCsvLine(line);
                    return {
                      rank: rankIdx >= 0 ? Number(cols[rankIdx]) : i + 1,
                      match_count: countIdx >= 0 ? Number(cols[countIdx]) : 0,
                      accuracy: accIdx >= 0 ? Number(cols[accIdx]) : 0,
                      pattern: patIdx >= 0 ? cols[patIdx] : "",
                    };
                  });
                })()
              : aggregateFromMatchedCsv(text);
          if (!cancelled && aggregated.length > 0) {
            setRows(aggregated);
            setSource(csvPath.split("/").pop() ?? csvPath);
          }
        }
      } catch {
        if (!cancelled) {
          setRows([]);
          setSource(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [runId, summaryJson, summaryCsv, matchedCsv]);

  if (loading) {
    return (
      <Card title="Most common matched patterns">
        <p className="text-sm text-brand-600">Loading pattern summary...</p>
      </Card>
    );
  }

  if (rows.length === 0) {
    return null;
  }

  return (
    <Card title="Most common matched patterns">
      <p className="text-sm text-brand-600 mb-4">
        Historical subtrace sequences BEST matched most often on the test set (from{" "}
        {source ?? "explainability output"}). Full sequences appear in the table; bar charts in
        Plots use shortened labels.
      </p>
      <div className="overflow-x-auto rounded-lg border border-brand-100">
        <table className="w-full text-sm">
          <thead className="bg-brand-50 text-brand-800">
            <tr>
              <th className="text-left px-3 py-2 font-medium w-12">#</th>
              <th className="text-left px-3 py-2 font-medium w-24">Matches</th>
              <th className="text-left px-3 py-2 font-medium w-24">Accuracy</th>
              <th className="text-left px-3 py-2 font-medium">Activity sequence</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.rank}-${row.pattern}`} className="border-t border-brand-100">
                <td className="px-3 py-2 text-brand-600">{row.rank}</td>
                <td className="px-3 py-2 font-medium">{row.match_count}</td>
                <td className="px-3 py-2">
                  <span
                    className={
                      row.accuracy >= 0.7
                        ? "text-green-700"
                        : row.accuracy >= 0.4
                        ? "text-amber-700"
                        : "text-red-700"
                    }
                  >
                    {(row.accuracy * 100).toFixed(0)}%
                  </span>
                </td>
                <td className="px-3 py-2 font-mono text-xs leading-relaxed break-words">
                  {row.pattern}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
