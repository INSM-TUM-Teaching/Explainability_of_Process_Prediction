import React, { useEffect, useState, useMemo } from "react";
import { ReactFlow, Controls, Background, useNodesState, useEdgesState, MarkerType } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from 'dagre';
import ProcessMapNode from "./ProcessMapNode";
import ProcessMapTerminalNode from "./ProcessMapTerminalNode";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from "recharts";

interface GlobalResultsProps {
  runId: string;
  datasetId: string;
  summary: any;
}

const nodeTypes = {
  activity: ProcessMapNode,
  start: ProcessMapTerminalNode,
  end: ProcessMapTerminalNode,
};

// Custom styles for highlight
const edgeStyles = `
  /* Bring selected edge to front */
  .react-flow__edge.selected {
    z-index: 9999 !important;
  }
  .react-flow__edge.selected .react-flow__edge-path {
    stroke: #2563eb !important;
    transition: all 0.2s ease;
    /* Create a high-contrast halo */
    filter: drop-shadow(0 0 3px #fff) drop-shadow(0 0 2px #fff);
  }
  /* Target the marker head color and size on selection */
  .react-flow__edge.selected marker path {
    fill: #2563eb !important;
    transition: fill 0.2s ease;
  }
  .react-flow__edge.selected .react-flow__edge-text {
    fill: #1d4ed8 !important;
    font-weight: 900 !important;
    font-size: 20px !important;
  }
  .react-flow__edge.selected .react-flow__edge-textbg {
    fill: #ffffff !important;
    stroke: #2563eb !important;
    stroke-width: 3px !important;
    fill-opacity: 1 !important;
  }
  .react-flow__edge-path {
    transition: stroke-width 0.2s ease, stroke 0.2s ease;
  }
  /* Optional: Fade other edges when one is selected */
  .react-flow__edges:has(.selected) .react-flow__edge:not(.selected) {
    opacity: 0.3;
    transition: opacity 0.3s ease;
  }
`;

const getLayoutedElements = (nodes: any[], edges: any[], direction = 'TB') => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));

  const nodeWidth = 180;
  const nodeHeight = 60;

  dagreGraph.setGraph({ rankdir: direction, ranksep: 120, nodesep: 150 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.targetPosition = direction === 'LR' ? 'left' : 'top';
    node.sourcePosition = direction === 'LR' ? 'right' : 'bottom';

    // We are shifting the dagre node position (which is center) to the top left
    // so it matches React Flow's expectation
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };

    return node;
  });

  return { nodes, edges };
};

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
            const initialNodes = pmData.nodes.map((n: any) => ({
              id: n.id,
              type: n.type === 'activity' ? 'activity' : (n.type === 'start' ? 'start' : 'end'),
              data: { label: n.label, count: n.count, type: n.type },
              position: { x: 0, y: 0 },
            }));

            // Calculate max weight for edge thickness scaling
            const maxWeight = Math.max(...pmData.edges.map((e: any) => e.weight), 1);

            const initialEdges = pmData.edges.map((e: any, idx: number) => {
              const isSmall = e.weight < (maxWeight * 0.1);
              return {
                id: `e${idx}`,
                source: e.source,
                target: e.target,
                label: e.weight.toLocaleString(),
                type: 'smoothstep',
                animated: false,
                style: {
                  strokeWidth: Math.max(2, (e.weight / maxWeight) * 8),
                  stroke: e.type === 'virtual' ? '#94a3b8' : '#334155',
                  strokeDasharray: e.type === 'virtual' ? '5,5' : 'none',
                },
                markerEnd: {
                  type: MarkerType.ArrowClosed,
                  color: e.type === 'virtual' ? '#94a3b8' : '#334155',
                  width: isSmall ? 35 : 25, // Specifically larger heads for the smallest arrows
                  height: isSmall ? 35 : 25,
                },
                labelStyle: { fill: '#0f172a', fontWeight: 900, fontSize: 18 }, // Even bigger uniform font
                labelBgPadding: [8, 6],
                labelBgBorderRadius: 6,
                labelBgStyle: { fill: '#ffffff', fillOpacity: 0.95, stroke: '#cbd5e1', strokeWidth: 1.5 },
                interactionWidth: 30,
                pathOptions: { borderRadius: 20 }, // Smoother rounded corners for 90-degree bends
              };
            });
            
            const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
              initialNodes,
              initialEdges
            );

            setNodes([...layoutedNodes]);
            setEdges([...layoutedEdges]);
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
      <style>{edgeStyles}</style>
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
          <div className="flex-1 border rounded bg-slate-50 relative">
            <ReactFlow 
              nodes={nodes} 
              edges={edges} 
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              fitView
              minZoom={0.2}
              elevateEdgesOnSelect={true}
            >
              <Background color="#cbd5e1" gap={20} />
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
