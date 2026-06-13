import React, { useEffect, useState } from "react";
import { ReactFlow, Controls, Background, useNodesState, useEdgesState } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from "recharts";

interface GlobalResultsProps {
  runId: string;
  datasetId: string;
  summary: any;
}

export default function GlobalResults({ runId, datasetId, summary }: GlobalResultsProps) {
  const [globalStats, setGlobalStats] = useState<any>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);
      try {
        // Fetch global stats
        const statsRes = await fetch(`http://localhost:8000/runs/${runId}/global`);
        if (!statsRes.ok) throw new Error("Failed to load global metrics");
        const statsData = await statsRes.json();
        setGlobalStats(statsData);

        // Fetch process map
        const pmRes = await fetch(`http://localhost:8000/datasets/${datasetId}/process_map`);
        if (!pmRes.ok) throw new Error("Failed to load process map");
        const pmData = await pmRes.json();

        if (pmData.error) {
           setError(pmData.error);
        } else {
            // Layout nodes simply (in a real app you'd use dagre or elkjs for proper directed graph layout)
            // For now, randomly distribute or grid layout
            const newNodes = pmData.nodes.map((n: any, idx: number) => ({
              id: n.id,
              data: { label: n.label },
              position: { x: (idx % 5) * 200, y: Math.floor(idx / 5) * 150 },
              style: {
                  border: '1px solid #784be8',
                  padding: '10px',
                  borderRadius: '5px',
                  background: 'white',
                  color: '#333'
              }
            }));
            const newEdges = pmData.edges.map((e: any, idx: number) => ({
              id: `e${idx}`,
              source: e.source,
              target: e.target,
              label: String(e.weight),
              animated: true,
            }));
            
            setNodes(newNodes);
            setEdges(newEdges);
        }
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [runId, datasetId]);

  if (loading) return <div className="p-8 text-center text-slate-500">Loading Global Data...</div>;
  if (error) return <div className="p-8 text-center text-red-500">Error: {error}</div>;
  if (!globalStats) return null;

  // Format data for charts
  const variantChartData = globalStats.variants.slice(0, 10).map((v: any) => ({
    name: v.variant.length > 20 ? v.variant.substring(0, 20) + "..." : v.variant,
    full_name: v.variant,
    accuracy: Number(v.accuracy.toFixed(2)),
    total: v.total_cases_in_test
  }));

  const prefixChartData = globalStats.prefix_accuracy.map((p: any) => ({
    prefix_length: `Length ${p.prefix_length}`,
    accuracy: Number(p.accuracy.toFixed(2)),
    total: p.total_cases
  }));

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* KPIs */}
      <div className="grid grid-cols-4 gap-4">
        <div className="rounded border bg-white p-4 text-center shadow-sm">
            <div className="text-sm text-slate-500 uppercase tracking-wide">Test Accuracy</div>
            <div className="mt-1 text-2xl font-semibold text-brand-700">{globalStats.overall_accuracy.toFixed(2)}%</div>
        </div>
        <div className="rounded border bg-white p-4 text-center shadow-sm">
            <div className="text-sm text-slate-500 uppercase tracking-wide">Total Cases</div>
            <div className="mt-1 text-2xl font-semibold text-brand-700">{summary?.dataset?.num_cases || "N/A"}</div>
        </div>
        <div className="rounded border bg-white p-4 text-center shadow-sm">
            <div className="text-sm text-slate-500 uppercase tracking-wide">Total Events</div>
            <div className="mt-1 text-2xl font-semibold text-brand-700">{summary?.dataset?.num_events || "N/A"}</div>
        </div>
        <div className="rounded border bg-white p-4 text-center shadow-sm">
            <div className="text-sm text-slate-500 uppercase tracking-wide">Unique Variants</div>
            <div className="mt-1 text-2xl font-semibold text-brand-700">{globalStats.unique_variants}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Variants Error Rate */}
          <div className="rounded border bg-white p-4 shadow-sm h-[400px]">
              <h3 className="text-md font-semibold mb-4 text-brand-900">Top 10 Variants - Accuracy vs Volume</h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={variantChartData} margin={{ top: 5, right: 30, left: 20, bottom: 50 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} interval={0} tick={{fontSize: 12}} />
                  <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                  <Tooltip />
                  <Legend verticalAlign="top" />
                  <Bar yAxisId="left" dataKey="total" name="Test Cases" fill="#8884d8" />
                  <Line yAxisId="right" type="monotone" dataKey="accuracy" name="Accuracy (%)" stroke="#82ca9d" strokeWidth={3} />
                </BarChart>
              </ResponsiveContainer>
          </div>

          {/* Accuracy by Prefix Length */}
          <div className="rounded border bg-white p-4 shadow-sm h-[400px]">
              <h3 className="text-md font-semibold mb-4 text-brand-900">Accuracy by Prefix Length</h3>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={prefixChartData} margin={{ top: 5, right: 30, left: 20, bottom: 50 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="prefix_length" angle={-45} textAnchor="end" height={60} tick={{fontSize: 12}} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Legend verticalAlign="top" />
                  <Line type="monotone" dataKey="accuracy" name="Accuracy (%)" stroke="#ff7300" strokeWidth={3} />
                </LineChart>
              </ResponsiveContainer>
          </div>
      </div>

      {/* Global Process Map */}
      <div className="rounded border bg-white p-4 shadow-sm h-[600px] flex flex-col">
          <h3 className="text-md font-semibold mb-2 text-brand-900">Global Process Map (Dataset Overview)</h3>
          <div className="flex-1 border rounded bg-slate-50">
            <ReactFlow 
              nodes={nodes} 
              edges={edges} 
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              fitView
            >
              <Background color="#ccc" gap={16} />
              <Controls />
            </ReactFlow>
          </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {globalStats.global_explanations && globalStats.global_explanations.shap_features && globalStats.global_explanations.shap_features.length > 0 && (
          <div className="rounded border bg-white p-4 shadow-sm h-[400px]">
            <h3 className="text-md font-semibold mb-2 text-brand-900">Global Feature Importance (SHAP)</h3>
            <p className="text-xs text-slate-500 mb-4">Averaged SHAP impact across a sample of {summary?.request?.config?.explainability_samples || 100} predictions.</p>
            <ResponsiveContainer width="100%" height="80%">
              <BarChart layout="vertical" data={globalStats.global_explanations.shap_features.sort((a: any, b: any) => b.importance - a.importance).slice(0, 15)} margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="feature" type="category" width={100} tick={{fontSize: 12}} />
                <Tooltip />
                <Bar dataKey="importance" fill="#8884d8" name="Mean Impact" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {globalStats.global_explanations && globalStats.global_explanations.lime_features && globalStats.global_explanations.lime_features.length > 0 && (
          <div className="rounded border bg-white p-4 shadow-sm h-[400px]">
            <h3 className="text-md font-semibold mb-2 text-brand-900">Global Feature Importance ({summary?.request?.model_type === "gnn" ? "GraphLIME" : "LIME"})</h3>
            <p className="text-xs text-slate-500 mb-4">Averaged {summary?.request?.model_type === "gnn" ? "GraphLIME" : "LIME"} weights across a sample of {summary?.request?.config?.explainability_samples || 100} predictions.</p>
            <ResponsiveContainer width="100%" height="80%">
              <BarChart layout="vertical" data={globalStats.global_explanations.lime_features.sort((a: any, b: any) => b.importance - a.importance).slice(0, 15)} margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="feature" type="category" width={100} tick={{fontSize: 12}} />
                <Tooltip />
                <Bar dataKey="importance" fill="#82ca9d" name="Mean Impact" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {globalStats.global_explanations && globalStats.global_explanations.gnn_features && globalStats.global_explanations.gnn_features.length > 0 && (
          <div className="rounded border bg-white p-4 shadow-sm h-[400px]">
            <h3 className="text-md font-semibold mb-2 text-brand-900">Global Feature Importance (Gradient)</h3>
            <p className="text-xs text-slate-500 mb-4">Averaged Gradient magnitude across a sample of {summary?.request?.config?.explainability_samples || 100} predictions.</p>
            <ResponsiveContainer width="100%" height="80%">
              <BarChart layout="vertical" data={globalStats.global_explanations.gnn_features.sort((a: any, b: any) => b.importance - a.importance).slice(0, 15)} margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="feature" type="category" width={100} tick={{fontSize: 12}} />
                <Tooltip />
                <Bar dataKey="importance" fill="#ffc658" name="Mean Impact" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
