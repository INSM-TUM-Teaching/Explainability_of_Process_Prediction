import React, { useEffect, useState, useMemo } from "react";
import Papa from "papaparse";
import { ReactFlow, Controls, Background, useNodesState, useEdgesState, MarkerType, Node, Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from 'dagre';
import { artifactUrl, API_BASE } from "../../lib/api";
import ProcessMapNode from "./ProcessMapNode";
import ProcessMapTerminalNode from "./ProcessMapTerminalNode";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Line } from "recharts";

interface GlobalResultsProps {
  runId: string;
  datasetId: string;
  summary: any;
  onCaseClick?: (cid: string) => void;
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

export default function GlobalResults({ runId, datasetId, summary, onCaseClick }: GlobalResultsProps) {
  const [globalStats, setGlobalStats] = useState<any>({
    overall_accuracy: 0,
    total_variants: 0,
    unique_variants: 0,
    variants: [],
    prefix_accuracy: [],
    global_explanations: {}
  });
  const [topPatterns, setTopPatterns] = useState<any[]>([]);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedEdge, setSelectedEdge] = useState<any>(null);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [expandedVariant, setExpandedVariant] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPattern, setSelectedPattern] = useState<any>(null);
  const [patternCases, setPatternCases] = useState<string[]>([]);
  const [loadingCases, setLoadingCases] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [sortConfig, setSortConfig] = useState<{ key: string, direction: 'asc' | 'desc' }>({ key: 'global_frequency', direction: 'desc' });

  const filteredPatterns = useMemo(() => {
    if (!searchQuery) return topPatterns;

    return topPatterns.filter(p => {
      // Format sequence the same way it's displayed in the table
      let seqStr = String(p.sequence);
      try {
        const parsed = JSON.parse(seqStr);
        if (Array.isArray(parsed)) {
          seqStr = parsed.join(" → ");
        }
      } catch (e) { }

      seqStr = seqStr.toLowerCase();
      const nextStr = String(p.predicted_next_activity).toLowerCase();
      const idStr = String(p.pattern_id).toLowerCase();
      const rawQuery = searchQuery.trim().toLowerCase();

      // 1. ID Search: if the query is just a number
      if (/^\d+$/.test(rawQuery)) {
        return idStr === rawQuery;
      }

      // 2. Exact Sequence Search: if the query is wrapped in parentheses
      if (rawQuery.startsWith("(") && rawQuery.endsWith(")")) {
        const innerQuery = rawQuery.slice(1, -1).trim();
        // Replace commas with arrows for comparison
        const queryAsSeq = innerQuery.replace(/,\s*/g, " → ");
        // Check for EXACT match of the entire sequence
        return seqStr === queryAsSeq;
      }

      // 3. Predicted Next Activity Search: default fallback for text
      return nextStr === rawQuery || nextStr.includes(rawQuery);
    });
  }, [topPatterns, searchQuery]);

  const sortedPatterns = useMemo(() => {
    let sortablePatterns = [...filteredPatterns];
    if (sortConfig !== null) {
      sortablePatterns.sort((a, b) => {
        let aValue = a[sortConfig.key];
        let bValue = b[sortConfig.key];

        if (sortConfig.key === 'pattern_id' || sortConfig.key === 'global_frequency') {
          aValue = parseInt(aValue, 10);
          bValue = parseInt(bValue, 10);
        } else if (sortConfig.key === 'global_accuracy') {
          aValue = parseFloat(aValue);
          bValue = parseFloat(bValue);
        }

        if (aValue < bValue) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (aValue > bValue) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }
    return sortablePatterns;
  }, [filteredPatterns, sortConfig]);

  const requestSort = (key: string) => {
    // Default direction depends on the column
    let direction: 'asc' | 'desc' = key === 'pattern_id' ? 'asc' : 'desc';
    
    // If clicking the same column, toggle the direction
    if (sortConfig && sortConfig.key === key) {
      direction = sortConfig.direction === 'asc' ? 'desc' : 'asc';
    }
    
    setSortConfig({ key, direction });
  };

  const getSortIcon = (key: string) => {
    if (sortConfig?.key !== key) return null;
    return sortConfig.direction === 'asc' ? <span className="text-brand-600 ml-1">▲</span> : <span className="text-brand-600 ml-1">▼</span>;
  };

  const handlePatternClick = async (pattern: any) => {
    setSelectedPattern(pattern);
    setIsModalOpen(true);
    setLoadingCases(true);
    setPatternCases([]);

    try {
      const pRes = await fetch(artifactUrl(runId, "best_predictions.csv"));
      if (pRes.ok) {
        const text = await pRes.text();
        const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });

        let pSeq: string[] = [];
        try { pSeq = JSON.parse(pattern.sequence); } catch (e) { }

        if (pSeq.length === 0) {
          setPatternCases([]);
          return;
        }

        const casesWithPattern = new Set<string>();

        (parsed.data as any[]).forEach(row => {
          if (!row.sequence) return;
          let cSeq: string[] = [];
          try { cSeq = JSON.parse(row.sequence); } catch (e) { return; }

          let hasMatch = false;
          for (let i = 0; i <= cSeq.length - pSeq.length; i++) {
            let match = true;
            for (let j = 0; j < pSeq.length; j++) {
              if (cSeq[i + j] !== pSeq[j]) {
                match = false;
                break;
              }
            }
            if (match) {
              hasMatch = true;
              break;
            }
          }
          if (hasMatch && row.case_id) {
            casesWithPattern.add(row.case_id);
          }
        });

        setPatternCases(Array.from(casesWithPattern));
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingCases(false);
    }
  };

  const onSelectionChange = React.useCallback(({ nodes: sNodes, edges: sEdges }: any) => {
    // Clear sidebar if nothing is selected
    if (sNodes.length === 0 && sEdges.length === 0) {
      setSelectedEdge(null);
      setSelectedNode(null);
      setExpandedVariant(null);
    }

    setEdges((eds) =>
      eds.map((e) => {
        const isSelected = sEdges.some((s: any) => s.id === e.id);
        const baseColor = (e.data as any)?.type === 'virtual' ? '#94a3b8' : '#334155';
        const markerEnd = e.markerEnd as any || {};
        return {
          ...e,
          markerEnd: {
            ...markerEnd,
            color: isSelected ? '#2563eb' : baseColor,
          },
        };
      })
    );
  }, [setEdges]);

  const onEdgeClick = React.useCallback((event: React.MouseEvent, edge: any) => {
    setSelectedEdge(edge);
    setSelectedNode(null);
    setExpandedVariant(null);
  }, []);

  const onNodeClick = React.useCallback((event: React.MouseEvent, node: any) => {
    if (node.type === 'activity') {
      setSelectedNode(node);
      setSelectedEdge(null);
      setExpandedVariant(null);
    }
  }, []);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);
      try {
        if (summary?.request?.model_type === "best") {
          // For best model, fetch top_patterns.csv
          try {
            const tpRes = await fetch(artifactUrl(runId, "explainability/top_patterns.csv"));
            if (tpRes.ok) {
              const text = await tpRes.text();
              const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
              const validPatterns = (parsed.data as any[]).filter(p => {
                try {
                  const seq = JSON.parse(p.sequence);
                  return Array.isArray(seq) && seq.length >= 2;
                } catch(e) {
                  return false;
                }
              });
              setTopPatterns(validPatterns);
            }
          } catch (e) { console.error("Could not load top patterns", e); }
        }

        // Fetch global stats for ALL models (including best)
        const statsRes = await fetch(`${API_BASE}/runs/${runId}/global`);
        if (!statsRes.ok) throw new Error("Failed to load global metrics");
        const statsData = await statsRes.json();
        setGlobalStats(statsData);

        // Fetch process map
        const pmRes = await fetch(`${API_BASE}/datasets/${datasetId}/process_map`);
        if (!pmRes.ok) throw new Error("Failed to load process map");
        const pmData = await pmRes.json();

        if (pmData.error) {
          setError(pmData.error);
        } else {
          const initialNodes = pmData.nodes.map((n: any) => ({
            id: n.id,
            type: n.type === 'activity' ? 'activity' : (n.type === 'start' ? 'start' : 'end'),
            data: { label: n.label, count: n.count, type: n.type, variants: n.variants },
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
              data: { type: e.type, variants: e.variants },
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
  let variantChartData = [];
  if (globalStats.variants && globalStats.variants.length > 0) {
    variantChartData = globalStats.variants.slice(0, 10).map((v: any) => ({
      name: `Variant ${v.id}`,
      full_name: v.variant,
      accuracy: Number(v.accuracy.toFixed(2)),
      total: v.total_cases_in_test
    }));
  } else if (topPatterns.length > 0) {
    // Top patterns used for variantChartData if best model
    variantChartData = topPatterns.slice(0, 10).map((p: any, idx: number) => ({
      name: `Variant ${p.pattern_id || (idx + 1)}`,
      full_name: p.sequence,
      accuracy: p.global_accuracy ? Number((parseFloat(p.global_accuracy) * 100).toFixed(2)) : 0,
      total: parseInt(p.global_frequency || "0", 10)
    }));
  }

  // Handle lowest accuracy variants
  let lowestAccuracyVariantsData: any[] = [];
  if (globalStats.variants && globalStats.variants.length > 0) {
    lowestAccuracyVariantsData = [...globalStats.variants]
      .sort((a: any, b: any) => a.accuracy - b.accuracy)
      .slice(0, 10)
      .map((v: any) => ({
        name: `Variant ${v.id}`,
        full_name: v.variant,
        accuracy: Number(v.accuracy.toFixed(2)),
        total: v.total_cases_in_test
      }));
  } else if (topPatterns.length > 0) {
    lowestAccuracyVariantsData = [...topPatterns]
      .map((p: any, idx: number) => ({
        name: `Variant ${p.pattern_id || (idx + 1)}`,
        full_name: p.sequence,
        accuracy: p.global_accuracy ? Number((parseFloat(p.global_accuracy) * 100).toFixed(2)) : 0,
        total: parseInt(p.global_frequency || "0", 10)
      }))
      .sort((a: any, b: any) => a.accuracy - b.accuracy)
      .slice(0, 10);
  }


  const testSetUniqueVariants = globalStats.variants?.length || 0;
  const testSetEvents = globalStats.total_test_events || (globalStats.variants || []).reduce((acc: number, v: any) => acc + v.total_cases_in_test, 0);
  const testSetCases = globalStats.total_test_cases || testSetEvents;

  return (
    <div className="flex flex-col gap-6 w-full">
      <style>{edgeStyles}</style>
      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-2">
        <div className="rounded border bg-white p-4 text-center shadow-sm">
          <div className="text-sm text-slate-500 uppercase tracking-wide">Total Events</div>
          <div className="mt-1 text-2xl font-semibold text-brand-700">{summary?.dataset?.num_events || "N/A"}</div>
        </div>
        <div className="rounded border bg-white p-4 text-center shadow-sm">
          <div className="text-sm text-slate-500 uppercase tracking-wide">Total Cases</div>
          <div className="mt-1 text-2xl font-semibold text-brand-700">{summary?.dataset?.num_cases || "N/A"}</div>
        </div>
        <div className="rounded border bg-white p-4 text-center shadow-sm">
          <div className="text-sm text-slate-500 uppercase tracking-wide">Unique Variants</div>
          <div className="mt-1 text-2xl font-semibold text-brand-700">{globalStats?.unique_variants || "N/A"}</div>
        </div>
      </div>

      {/* Global Process Map */}
      <div className="rounded border bg-white p-4 shadow-sm h-[700px] flex flex-col">
        <h2 className="text-xl font-bold mb-4 text-brand-900 border-b pb-2">Global Process Map (Dataset Overview)</h2>
        <div className="flex-1 flex gap-4 overflow-hidden">
          <div className="flex-1 border rounded bg-slate-50 relative">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onSelectionChange={onSelectionChange}
              onEdgeClick={onEdgeClick}
              onNodeClick={onNodeClick}
              fitView
              minZoom={0.1}
              elevateEdgesOnSelect={true}
            >
              <Background color="#cbd5e1" gap={20} />
              <Controls />
            </ReactFlow>
          </div>

          {/* Sidebar for Variants */}
          {(selectedEdge || selectedNode) && (
            <div className="w-80 border rounded bg-white flex flex-col shadow-sm animate-in slide-in-from-right duration-300">
              <div className="p-3 border-b bg-slate-50">
                <div className="flex justify-between items-start mb-1">
                  <h4 className="text-sm font-bold text-slate-800">
                    {selectedEdge ? "Transition Variants" : "Activity Variants"}
                  </h4>
                  <button
                    onClick={() => {
                      setSelectedEdge(null);
                      setSelectedNode(null);
                      setExpandedVariant(null);
                    }}
                    className="text-slate-400 hover:text-slate-600 p-1"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                  </button>
                </div>
                <div className="text-[11px] text-slate-500 font-medium break-all">
                  {selectedEdge ? (
                    <>{selectedEdge.source} <span className="text-brand-500 mx-1">→</span> {selectedEdge.target}</>
                  ) : (
                    <span className="text-brand-600 font-bold px-1.5 py-0.5 bg-brand-50 rounded border border-brand-100">{selectedNode.data.label}</span>
                  )}
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-3">
                <div className="space-y-3">
                  {((selectedEdge?.data?.variants) || (selectedNode?.data?.variants))?.length > 0 ? (
                    ((selectedEdge?.data?.variants) || (selectedNode?.data?.variants)).map((v: any, idx: number) => {
                      const isExpanded = expandedVariant === idx;
                      return (
                        <div
                          key={idx}
                          className={`p-3 rounded-lg border transition-all cursor-pointer ${isExpanded ? 'border-brand-300 bg-brand-50/30' : 'border-slate-100 bg-slate-50/50 hover:border-brand-200 hover:bg-white'
                            }`}
                          onClick={() => setExpandedVariant(isExpanded ? null : idx)}
                        >
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-[10px] font-bold text-brand-600 bg-brand-50 px-2 py-0.5 rounded-full border border-brand-100">
                              {v.count} case(s)
                            </span>
                            <div className="flex items-center gap-1.5">
                              <span className="text-[9px] text-slate-400 font-mono font-bold uppercase">Variant {v.id}</span>
                              <svg
                                className={`transition-transform duration-200 text-slate-400 ${isExpanded ? 'rotate-180' : ''}`}
                                width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"
                              >
                                <polyline points="6 9 12 15 18 9"></polyline>
                              </svg>
                            </div>
                          </div>

                          <div className={`text-[11px] leading-relaxed text-slate-600 font-medium ${isExpanded ? '' : 'line-clamp-2'}`}>
                            {v.signature.split(" -> ").map((step: string, sIdx: number, arr: string[]) => {
                              let isHighlighted = false;
                              if (selectedEdge) {
                                const isStartTransition = selectedEdge.source === '__START__';
                                const isSource = !isStartTransition && step === selectedEdge.source;
                                const isTarget = (isStartTransition && sIdx === 0 && step === selectedEdge.target) ||
                                  (step === selectedEdge.target && sIdx > 0 && arr[sIdx - 1] === selectedEdge.source);
                                isHighlighted = isSource || isTarget;
                              } else if (selectedNode) {
                                isHighlighted = step === selectedNode.data.label;
                              }

                              return (
                                <React.Fragment key={sIdx}>
                                  <span className={isHighlighted ? "text-brand-600 font-bold bg-brand-50 rounded px-0.5 border border-brand-100" : ""}>
                                    {step}
                                  </span>
                                  {sIdx < arr.length - 1 && <span className="text-slate-300 mx-1">→</span>}
                                </React.Fragment>
                              );
                            })}
                          </div>

                          {/* Case IDs List (Visible when expanded) */}
                          {isExpanded && v.cases && (
                            <div className="mt-3 pt-3 border-t border-brand-100 animate-in fade-in zoom-in-95 duration-200">
                              <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider mb-2">Contained Cases:</div>
                              <div className="max-h-32 overflow-y-auto pr-1 custom-scrollbar">
                                <div className="flex flex-wrap gap-1.5">
                                  {v.cases.map((cId: string, cIdx: number) => (
                                    <span 
                                      key={cIdx} 
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        onCaseClick && onCaseClick(cId);
                                      }}
                                      className="px-2 py-0.5 bg-white border border-slate-200 rounded text-[10px] text-slate-600 font-mono font-medium cursor-pointer hover:bg-brand-50 hover:border-brand-200 hover:text-brand-700 transition-colors"
                                    >
                                      {cId}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-center py-8">
                      <div className="text-slate-300 mb-2">No variants found</div>
                      <p className="text-[11px] text-slate-400">This could be a virtual transition showing start/end frequencies.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Prediction Results Subsection */}
      <div className="flex flex-col gap-4 mt-4">
        <h2 className="text-xl font-bold text-brand-900 border-b pb-2">Prediction Results (for test set)</h2>
        
        {/* Prediction Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-2">
          <div className="rounded border bg-brand-50 border-brand-200 p-4 text-center shadow-sm">
              <div className="text-sm text-brand-600 uppercase tracking-wide font-semibold">Test Accuracy</div>
              <div className="mt-1 text-2xl font-bold text-brand-800">{globalStats.overall_accuracy.toFixed(2)}%</div>
          </div>
          <div className="rounded border bg-white p-4 text-center shadow-sm">
              <div className="text-sm text-slate-500 uppercase tracking-wide">Test Set Events</div>
              <div className="mt-1 text-2xl font-semibold text-slate-700">{testSetEvents}</div>
          </div>
          <div className="rounded border bg-white p-4 text-center shadow-sm">
              <div className="text-sm text-slate-500 uppercase tracking-wide">Test Set Cases</div>
              <div className="mt-1 text-2xl font-semibold text-slate-700">{testSetCases}</div>
          </div>
          <div className="rounded border bg-white p-4 text-center shadow-sm">
              <div className="text-sm text-slate-500 uppercase tracking-wide">Unique Variants</div>
              <div className="mt-1 text-2xl font-semibold text-slate-700">{testSetUniqueVariants}</div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
        {/* Top Variants Error Rate */}
        <div className="rounded border bg-white p-4 shadow-sm h-[400px]">
          <h3 className="text-md font-semibold mb-4 text-brand-900">Top 10 Variants - Accuracy vs Volume</h3>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={variantChartData} margin={{ top: 5, right: 30, left: 20, bottom: 50 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} interval={0} tick={{ fontSize: 12 }} />
              <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
              <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
              <Tooltip />
              <Legend verticalAlign="top" />
              <Bar yAxisId="left" dataKey="total" name="Test Cases" fill="#8884d8" />
              <Line yAxisId="right" type="monotone" dataKey="accuracy" name="Accuracy (%)" stroke="#82ca9d" strokeWidth={3} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Top 10 Variants with Lowest Accuracy */}
        <div className="rounded border bg-white p-4 shadow-sm h-[400px]">
          <h3 className="text-md font-semibold mb-4 text-brand-900">Top 10 Variants with Lowest Accuracy</h3>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={lowestAccuracyVariantsData} margin={{ top: 5, right: 30, left: 20, bottom: 50 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} interval={0} tick={{ fontSize: 12 }} />
              <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
              <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
              <Tooltip />
              <Legend verticalAlign="top" />
              <Bar yAxisId="left" dataKey="total" name="Test Cases" fill="#8884d8" />
              <Line yAxisId="right" type="monotone" dataKey="accuracy" name="Accuracy (%)" stroke="#82ca9d" strokeWidth={3} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
      </div>
      
      {/* Explainability Results Subsection */}
      {(globalStats.global_explanations?.shap_features?.length > 0 || globalStats.global_explanations?.lime_features?.length > 0 || globalStats.global_explanations?.gnn_features?.length > 0 || (summary?.request?.model_type === "best" && topPatterns.length > 0)) && (
        <div className="flex flex-col gap-4 mt-4">
          <h2 className="text-xl font-bold text-brand-900 border-b pb-2">Explainability Results (for test set)</h2>
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
                <Bar dataKey="importance" fill="#8884d8" name="Mean Impact" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        </div>
        
        {/* Top Patterns Table (for BEST model) */}
        {summary?.request?.model_type === "best" && topPatterns.length > 0 && (
          <div className="rounded border bg-white p-4 shadow-sm mt-2">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-md font-semibold text-brand-900">Top 20 Patterns</h3>
            <input
              type="text"
              placeholder="Search: ID Number, Predicted Activity, or (A, B, C) for Exact Sequence..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="border rounded px-3 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-brand-500 w-[500px]"
            />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-slate-50">
                  <th className="text-left py-2 px-3 font-semibold text-slate-700 cursor-pointer select-none hover:bg-slate-100 whitespace-nowrap" onClick={() => requestSort('pattern_id')}>
                    ID {getSortIcon('pattern_id')}
                  </th>
                  <th className="text-left py-2 px-3 font-semibold text-slate-700">Sequence</th>
                  <th className="text-left py-2 px-3 font-semibold text-slate-700 whitespace-nowrap">Predicted Activity</th>
                  <th className="text-center py-2 px-3 font-semibold text-slate-700 cursor-pointer select-none hover:bg-slate-100 whitespace-nowrap" onClick={() => requestSort('global_frequency')}>
                    Frequency {getSortIcon('global_frequency')}
                  </th>
                  <th className="text-center py-2 px-3 font-semibold text-slate-700 cursor-pointer select-none hover:bg-slate-100 whitespace-nowrap" onClick={() => requestSort('global_accuracy')}>
                    Accuracy {getSortIcon('global_accuracy')}
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedPatterns.slice(0, 20).map((pattern: any, idx: number) => {
                  let sequence = pattern.sequence;
                  try {
                    sequence = JSON.parse(pattern.sequence);
                    if (Array.isArray(sequence)) {
                      sequence = sequence.join(" → ");
                    }
                  } catch (e) {
                    // Keep original if parsing fails
                  }

                  const accuracy = parseFloat(pattern.global_accuracy) > 1
                    ? parseFloat(pattern.global_accuracy).toFixed(2)
                    : (parseFloat(pattern.global_accuracy) * 100).toFixed(2);

                  return (
                    <tr
                      key={idx}
                      className="border-b hover:bg-brand-50 cursor-pointer transition-colors"
                      onClick={() => handlePatternClick(pattern)}
                      title="Click to see cases with this pattern"
                    >
                      <td className="py-2 px-3 font-mono text-slate-600">{pattern.pattern_id}</td>
                      <td className="py-2 px-3 text-slate-700 font-medium text-xs">{sequence}</td>
                      <td className="py-2 px-3">
                        <span className="inline-block px-2 py-1 rounded bg-brand-50 border border-brand-100 text-brand-700 font-medium">
                          {pattern.predicted_next_activity}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-center text-slate-600">{parseInt(pattern.global_frequency || "0", 10).toLocaleString()}</td>
                      <td className="py-2 px-3 text-center">
                        <span className="text-green-600 font-semibold">{accuracy}%</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {isModalOpen && selectedPattern && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50" onClick={() => setIsModalOpen(false)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[80vh] flex flex-col shadow-xl" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-brand-900">
                Cases for Pattern {selectedPattern.pattern_id}
              </h3>
              <button
                onClick={() => setIsModalOpen(false)}
                className="text-slate-400 hover:text-slate-600 text-xl font-bold"
              >
                ✕
              </button>
            </div>

            <div className="mb-4 p-3 bg-slate-50 rounded border text-sm">
              <div className="font-semibold text-slate-700 mb-1">Sequence:</div>
              <div className="font-mono text-slate-600">{(() => {
                try {
                  const parsed = JSON.parse(selectedPattern.sequence);
                  if (Array.isArray(parsed)) return parsed.join(" → ");
                } catch (e) { }
                return selectedPattern.sequence;
              })()}</div>
              <div className="mt-2 text-brand-700 font-medium">Predicts: {selectedPattern.predicted_next_activity}</div>
            </div>

            <h4 className="font-medium text-slate-700 mb-2">Matching Cases:</h4>

            <div className="flex-1 overflow-y-auto border rounded bg-white p-4">
              {loadingCases ? (
                <div className="flex items-center justify-center py-8">
                  <div className="w-6 h-6 border-2 border-brand-600 border-t-transparent rounded-full animate-spin mr-2"></div>
                  <span className="text-slate-500 font-medium">Scanning predictions...</span>
                </div>
              ) : patternCases.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {patternCases.map(cid => (
                    <span
                      key={cid}
                      onClick={() => {
                        setIsModalOpen(false);
                        onCaseClick && onCaseClick(cid);
                      }}
                      className="px-3 py-1.5 bg-brand-50 hover:bg-brand-100 text-brand-700 text-sm font-medium rounded border border-brand-200 shadow-sm cursor-pointer transition-colors"
                    >
                      {String(cid).startsWith("Case") ? cid : `Case ${cid}`}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-slate-500 bg-slate-50 rounded border border-dashed">
                  No cases found containing this pattern.
                </div>
              )}
            </div>

            <div className="mt-4 pt-3 border-t text-right">
              <button
                className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white font-medium rounded-md shadow-sm transition-colors"
                onClick={() => setIsModalOpen(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
      )}
    </div>
  );
}
