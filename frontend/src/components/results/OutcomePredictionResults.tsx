import { useEffect, useState } from "react";
import { artifactUrl, artifactsZipUrl } from "../../lib/api";
import { explainOnDemand } from "../../lib/api_explain";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import GlobalResults from "./GlobalResults";
import Papa from "papaparse";

type PredictionRecord = {
  case_id: string;
  case_index: number;
  sequence: string;
  true_next_activity: string;
  predicted_next_activity: string;
  confidence_percent?: number;
  confidence?: number;
  correct?: number;
  variant_id?: string | number;
};

export default function OutcomePredictionResults({ runId, summary, uploadedFileName, configMode }: any) {
  const [predictions, setPredictions] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [isExactSearch, setIsExactSearch] = useState(false);
  const [activeTab, setActiveTab] = useState("global");
  
  useEffect(() => {
    async function loadData() {
      let predFile = "transformer_predictions.json";
      if (summary.request.model_type === "gnn") {
        predFile = "gnn_predictions.json";
      } else if (summary.request.model_type === "best") {
        predFile = "best_predictions.csv";
      }
      
      try {
        const res = await fetch(artifactUrl(runId, predFile));
        if (res.ok) {
          if (predFile.endsWith(".csv")) {
            // Parse CSV for best model
            const text = await res.text();
            const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
            const validData = (parsed.data as any[])
              .filter(row => row.case_id && row.case_index)
              .map(row => ({
                ...row,
                case_id: String(row.case_id).replace(/^Case\s+/i, ""),
                case_index: parseInt(String(row.case_index), 10),
                confidence: parseFloat(String(row.confidence)) || 0,
                true_next_activity: row.true_outcome || row.actual_outcome || row.true_next_activity,
                predicted_next_activity: row.predicted_outcome || row.predicted_next_activity
              }));
            setPredictions(validData);
          } else {
            // Parse JSON for transformer/gnn
            const data = await res.json();
            setPredictions(data);
          }
        }
      } catch (err) {
        console.error("Failed to load predictions", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [runId, summary]);


  const caseIds = Array.from(new Set(predictions.map(p => p.case_id))).sort();
  
  const filteredCases = caseIds.filter(cid => {
    const caseRecs = predictions.filter(p => p.case_id === String(cid));
    const vId = String(caseRecs[0]?.variant_id || "");
    const searchTerm = String(search).toLowerCase().replace(/^case\s+/i, '').trim();
    
    if (isExactSearch && searchTerm) {
      return String(cid).toLowerCase() === searchTerm || vId.toLowerCase() === searchTerm;
    }
    
    return String(cid).toLowerCase().includes(searchTerm) || vId.toLowerCase().includes(searchTerm);
  });

  const config = summary.request.config || {};
  const metrics = summary.metrics || {};
  
  const handleCaseClick = (caseId: string) => {
    setActiveTab("local");
    const cleanCaseId = String(caseId).replace(/^case\s+/i, '').trim();
    setSearch(cleanCaseId);
    setIsExactSearch(true);
  };

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Top Section */}
      <div className="flex flex-col gap-4 rounded-2xl border border-brand-100 bg-white p-6 shadow-sm md:flex-row md:items-center md:justify-between">
        <div className="flex-1 min-w-0 pr-4">
          <div className="text-sm font-medium uppercase tracking-[0.18em] text-brand-500">
            Prediction and Explainability Results
          </div>
          <h1 className="mt-2 text-2xl font-semibold text-brand-900">Run {runId}</h1>
          <div className="mt-3 text-sm text-slate-600 flex flex-wrap items-center gap-x-3 gap-y-1">
            <span>Uploaded file: <strong>{uploadedFileName || summary.dataset?.filename || "Unknown"}</strong></span>
            <span className="text-slate-300">|</span>
            <span>Model: <strong className="capitalize">{summary.request?.model_type || "Unknown"}</strong></span>
            <span className="text-slate-300">|</span>
            <span>Prediction Task: <strong className="capitalize">{summary.request?.task?.replace('_', ' ') || "Outcome Prediction"}</strong></span>
            <span className="text-slate-300">|</span>
            <span>Configuration: <strong className="capitalize">{configMode || "default"}</strong></span>
            <span className="text-slate-300">|</span>
            <span>Explainability: <strong>
              {summary.request?.explainability === "all"
                ? (summary.request?.model_type === "transformer" ? "SHAP and LIME" : "Gradient and GraphLIME")
                : (summary.request?.explainability === "graphlime" ? "GraphLIME" 
                   : (summary.request?.explainability === "pattern_analysis" ? "Pattern Analysis"
                   : String(summary.request?.explainability || "None").toUpperCase()))}
            </strong></span>
          </div>
        </div>
        <div className="flex gap-3 mt-4 md:mt-0 items-center shrink-0">
          <a href={artifactsZipUrl(runId)} className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700">
            Download
          </a>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full mt-4">
        <TabsList className="mb-6 h-auto p-1.5 bg-slate-100 border border-slate-200 rounded-xl shadow-inner inline-flex">
          <TabsTrigger 
            value="global"
            className="text-base px-8 py-2.5 rounded-lg data-[state=active]:bg-white data-[state=active]:text-brand-700 data-[state=active]:shadow-md data-[state=active]:font-bold text-slate-500 font-medium transition-all duration-300 hover:text-brand-600"
          >
            Global Overview
          </TabsTrigger>
          <TabsTrigger 
            value="local"
            className="text-base px-8 py-2.5 rounded-lg data-[state=active]:bg-white data-[state=active]:text-brand-700 data-[state=active]:shadow-md data-[state=active]:font-bold text-slate-500 font-medium transition-all duration-300 hover:text-brand-600"
          >
            Local Case Analysis
          </TabsTrigger>
        </TabsList>
        <TabsContent value="global">
          <GlobalResults runId={runId} datasetId={summary.dataset?.dataset_id} summary={summary} onCaseClick={handleCaseClick} />
        </TabsContent>
        <TabsContent value="local">
          <div className="rounded-2xl border border-brand-100 bg-white p-6 shadow-sm mt-4">
            <div className="flex items-center justify-between gap-3 mb-6">
              <h2 className="text-lg font-semibold text-brand-900">Local Prediction Data</h2>
              <input 
                type="text" 
                placeholder="Search by Case ID or Variant ID..." 
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setIsExactSearch(false);
                }}
                className="border rounded px-3 py-1 text-sm w-72"
              />
            </div>

            {loading ? (
              <div>Loading prediction data...</div>
            ) : filteredCases.length === 0 ? (
              <div className="text-slate-500">No cases or variants found in test set {search && `matching "${search}"`}</div>
            ) : (
              <div className="flex flex-col gap-4">
                <div className="text-sm text-slate-600">Showing {filteredCases.length} of {caseIds.length} cases</div>
                {filteredCases.map(cid => (
                  <CasePredictionBlock 
                    key={cid} 
                    caseId={cid} 
                    records={predictions.filter(p => p.case_id === cid)} 
                    runId={runId}
                    modelType={summary.request?.model_type}
                    taskType="outcome"
                    autoExpand={search === cid}
                    explainabilityType={summary.request?.explainability || "none"}
                  />
                ))}
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}


function CasePredictionBlock({ caseId, records, runId, modelType, taskType = "outcome", explainabilityType, autoExpand = false }: { caseId: string, records: any[], runId: string, modelType: string, taskType?: string, explainabilityType?: string, autoExpand?: boolean }) {
  const [expanded, setExpanded] = useState(autoExpand);
  const maxIndex = Math.max(...records.map(r => r.case_index));

  useEffect(() => {
    if (autoExpand) {
      setExpanded(true);
    }
  }, [autoExpand]);
  const [selectedIndex, setSelectedIndex] = useState(maxIndex);

  const selectedRecord = records.find(r => r.case_index === selectedIndex) || records[0];
  const variantId = selectedRecord?.variant_id;

  const isBestModel = modelType === "best";
  const explains = isBestModel ? ["Pattern"] : (modelType === "transformer" ? ["SHAP", "LIME"] : ["Gradient", "GraphLIME"]);
  const [selectedExplain, setSelectedExplain] = useState(explains[0]);
  const [explainResult, setExplainResult] = useState<any>(null);
  const [explaining, setExplaining] = useState(false);

  // When expanding or changing index, clear old explanation
  useEffect(() => {
    setExplainResult(null);
  }, [expanded, selectedIndex]);

  const handleExplain = async () => {
    if (isBestModel) {
      // Load pattern analysis and build heatmap data
      setExplaining(true);
      try {
        const patRes = await fetch(
          artifactUrl(runId, "explainability/pattern_analysis.json")
        );
        if (patRes.ok) {
          const patData = await patRes.json();
          
          // Load top patterns to get pattern info
          const topRes = await fetch(
            artifactUrl(runId, "explainability/top_patterns.csv")
          );
          let patternMap: any = {};
          if (topRes.ok) {
            const text = await topRes.text();
            const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
            (parsed.data as any[]).forEach((p: any) => {
              try {
                const seq = JSON.parse(p.sequence);
                if (Array.isArray(seq) && seq.length >= 2) {
                  patternMap[p.pattern_id] = {
                    sequence: p.sequence,
                    predicted_next: p.predicted_next_activity,
                    frequency: parseInt(p.global_frequency) || 0
                  };
                }
              } catch(e) {}
            });
          }

          // Get the pattern analysis for this case/index
          let caseKey = `case_${caseId}`;
          let caseData = patData[caseKey];
          if (!caseData) {
            caseData = patData[caseId] || {};
          }
          const indexKey = `index_${selectedIndex}`;
          const indexData = caseData[indexKey] || { full_sequence: [], all_pattern_matches: [] };
          
          let prefixSeq: string[] = [];
          if (indexData.full_sequence && indexData.full_sequence.length > 0) {
            prefixSeq = indexData.full_sequence;
          } else {
            try { prefixSeq = JSON.parse(selectedRecord.sequence); } catch(e) {}
          }
          
          // Dynamically compute matches against all top patterns
          const dynamicMatches: any[] = [];
          Object.keys(patternMap).forEach(patternId => {
            const p = patternMap[patternId];
            let pSeq: string[] = [];
            try { pSeq = JSON.parse(p.sequence); } catch(e) { return; }
            if (pSeq.length === 0) return;
            
            for (let i = 0; i <= prefixSeq.length - pSeq.length; i++) {
              let match = true;
              for (let j = 0; j < pSeq.length; j++) {
                if (prefixSeq[i+j] !== pSeq[j]) {
                  match = false;
                  break;
                }
              }
              if (match) {
                dynamicMatches.push({
                  pattern_id: patternId,
                  start_offset: i,
                  end_offset: i + pSeq.length - 1,
                  frequency: p.frequency
                });
              }
            }
          });
          
          setExplainResult({
            type: "pattern_heatmap",
            sequence: prefixSeq,
            matches: dynamicMatches,
            patternMap: patternMap
          });
        }
      } catch (e) {
        console.error("Failed to load pattern analysis", e);
        alert("Failed to load pattern analysis");
      } finally {
        setExplaining(false);
      }
    } else {
      // Original explanation logic for GNN/Transformer
      setExplaining(true);
      try {
        await explainOnDemand(runId, caseId, selectedIndex, selectedExplain);
        setExplainResult({ method: selectedExplain, path: `explainability/${caseId}_${selectedIndex}`, timestamp: Date.now() });
      } catch(e) {
        console.error(e);
        alert("Explanation failed");
      } finally {
        setExplaining(false);
      }
    }
  };

  if (!selectedRecord) return null;

  const isCorrect = selectedRecord.true_next_activity === selectedRecord.predicted_next_activity;

  return (
    <div className="border rounded-lg overflow-hidden">
      <div 
        className="bg-slate-50 px-4 py-3 cursor-pointer flex justify-between items-center hover:bg-slate-100"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-4">
          <div className="font-semibold">Case ID: {caseId}</div>
          {variantId && (
            <div className="px-2 py-0.5 bg-brand-50 border border-brand-200 text-brand-700 text-[10px] font-bold rounded uppercase tracking-wider">
              Variant: {variantId}
            </div>
          )}
        </div>
        <div className="flex items-center gap-4 text-sm" onClick={e => e.stopPropagation()}>
          <label>Index:</label>
          <select 
            value={selectedIndex} 
            onChange={e => setSelectedIndex(Number(e.target.value))}
            className="border rounded px-2 py-1 bg-white"
          >
            {records.map(r => (
              <option key={r.case_index} value={r.case_index}>{r.case_index}</option>
            ))}
          </select>
        </div>
      </div>
      
      {expanded && (
        <div className="p-4 bg-white border-t text-sm flex flex-col gap-4">
          <div>
            <strong>Trace History / Sequence:</strong>
            <div className="mt-2 flex flex-wrap items-center gap-2 overflow-x-auto pb-2">
              {/* Historical Sequence */}
              {(() => {
                let activities: string[] = [];
                try {
                  activities = JSON.parse(selectedRecord.sequence);
                } catch {
                  activities = selectedRecord.sequence.split(',').map((a: string) => a.trim());
                }
                return activities.map((act: string, i: number, arr: string[]) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className="bg-brand-50 border border-brand-500 text-brand-800 px-3 py-1.5 rounded-md shadow-sm text-xs font-medium whitespace-nowrap">
                      {act.trim()}
                    </div>
                    {/* Arrow: Solid for history, Longer Dashed for the transition to target */}
                    {i === arr.length - 1 ? (
                      <svg className="w-10 h-4 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 40 24">
                        {/* The Stem - Dashed */}
                        <path 
                          strokeLinecap="round" 
                          strokeLinejoin="round" 
                          strokeWidth={2} 
                          strokeDasharray="6 6"
                          d="M3 12h34" 
                        />
                        {/* The Head - Solid */}
                        <path 
                          strokeLinecap="round" 
                          strokeLinejoin="round" 
                          strokeWidth={2} 
                          d="M30 5l7 7-7 7" 
                        />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path 
                          strokeLinecap="round" 
                          strokeLinejoin="round" 
                          strokeWidth={2} 
                          d="M14 5l7 7-7 7M21 12H3" 
                        />
                      </svg>
                    )}
                  </div>
                ));
              })()}
              
              {/* True Outcome (Target) */}
              <div className="flex items-center gap-2">
                <div className="bg-slate-50 border border-slate-400 border-dashed text-slate-600 px-3 py-1.5 rounded-md text-xs font-bold whitespace-nowrap">
                  {selectedRecord.true_next_activity}
                </div>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4">
            <div className="p-3 border rounded">
              <div className="text-slate-500 mb-1">
                True Outcome
              </div>
              <div className="font-semibold">{selectedRecord.true_next_activity}</div>
            </div>
            <div className={`p-3 border rounded ${isCorrect ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
              <div className={`mb-1 ${isCorrect ? 'text-green-700' : 'text-red-700'}`}>
                Predicted Outcome
              </div>
              <div className={`font-semibold ${isCorrect ? 'text-green-800' : 'text-red-800'}`}>{selectedRecord.predicted_next_activity}</div>
            </div>
            <div className="p-3 border rounded">
              <div className="text-slate-500 mb-1">Confidence</div>
              <div className="font-semibold">
                {(() => {
                  const conf = selectedRecord.confidence_percent ?? selectedRecord.confidence;
                  if (typeof conf === 'string') {
                    // Handle "NaN" strings
                    const parsed = parseFloat(conf);
                    if (isNaN(parsed)) return "N/A";
                    // Check if it's already a percentage (>1) or 0-1 range
                    return parsed > 1 ? `${parsed.toFixed(2)}%` : `${(parsed * 100).toFixed(2)}%`;
                  }
                  if (typeof conf === 'number') {
                    if (isNaN(conf)) return "N/A";
                    return conf > 1 ? `${conf.toFixed(2)}%` : `${(conf * 100).toFixed(2)}%`;
                  }
                  return "N/A";
                })()}
              </div>
            </div>
          </div>

          {explainabilityType !== "none" && (
            <div className="mt-4 pt-4 border-t flex flex-col gap-3">
              <div className="flex gap-3 items-center">
                <strong>{isBestModel ? "Pattern Analysis:" : "Generate Explanation:"}</strong>
                {!isBestModel && (
                  <select className="border rounded px-2 py-1" value={selectedExplain} onChange={e => setSelectedExplain(e.target.value)}>
                    {explains.map(ex => <option key={ex} value={ex}>{ex}</option>)}
                  </select>
                )}
                <button 
                  onClick={handleExplain} 
                  disabled={explaining}
                  className="bg-brand-600 text-white px-4 py-1.5 rounded text-sm hover:bg-brand-700 disabled:opacity-50"
                >
                  {explaining ? "Generating..." : "Generate"}
                </button>
              </div>
            
            {explainResult && (
              <div className="mt-4 border rounded p-4 bg-slate-50">
                {explainResult.type === "pattern_heatmap" ? (
                  <div>
                    <h4 className="font-medium mb-3">Pattern Heatmap</h4>
                    <p className="text-xs text-slate-500 mb-3">Darker blue = more prominent patterns | Lighter = less common patterns</p>
                    <div className="flex flex-wrap items-center gap-2 pb-4">
                      {(explainResult.sequence as string[]).map((activity, idx) => {
                        // Calculate intensity for this position based on matching patterns
                        const matchesAtPosition = (explainResult.matches as any[]).filter(m => 
                          idx >= m.start_offset && idx <= m.end_offset
                        );
                        
                        if (matchesAtPosition.length === 0) {
                          // No patterns - light gray
                          return (
                            <div
                              key={idx}
                              className="px-3 py-1.5 rounded-md text-xs font-medium whitespace-nowrap text-slate-600 border border-slate-300"
                              style={{ backgroundColor: "rgba(226, 232, 240, 0.7)" }}
                              title="No patterns"
                            >
                              {activity}
                            </div>
                          );
                        }
                        
                        // Use the maximum frequency of patterns at this position instead of the average
                        // so that a highly prominent pattern isn't diluted by overlapping rare patterns
                        const maxFreqAtPos = Math.max(...matchesAtPosition.map(m => m.frequency));
                        
                        // Get global max frequency for normalization (across this case)
                        const allFrequencies = (explainResult.matches as any[]).map(m => m.frequency);
                        const globalMax = Math.max(...allFrequencies, 1);
                        
                        // Normalize to 0-1 range
                        const normalizedIntensity = maxFreqAtPos / globalMax;
                        
                        // Create a color gradient: light blue for low, dark blue for high
                        const hue = 217;
                        const saturation = 100;
                        const lightness = 100 - (normalizedIntensity * 60);
                        const bgColor = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                        const textColor = normalizedIntensity > 0.6 ? "white" : "rgb(15, 23, 42)";

                        const matchedPatternNames = matchesAtPosition.map(m => {
                          const pInfo = explainResult.patternMap[m.pattern_id];
                          let seqStr = pInfo ? pInfo.sequence : "";
                          try {
                            const parsed = JSON.parse(seqStr);
                            if (Array.isArray(parsed)) seqStr = parsed.join(" → ");
                          } catch(e) {}
                          return pInfo ? `Pattern ${m.pattern_id}: ${seqStr} → ${pInfo.predicted_next}` : `Pattern ${m.pattern_id}`;
                        }).join("\n");

                        return (
                          <div
                            key={idx}
                            className="px-3 py-1.5 rounded-md text-xs font-medium whitespace-nowrap border border-slate-300 transition-colors"
                            style={{
                              backgroundColor: bgColor,
                              color: textColor
                            }}
                            title={`${matchesAtPosition.length} pattern(s), max frequency: ${maxFreqAtPos.toFixed(1)}\n${matchedPatternNames}`}
                          >
                            {activity}
                          </div>
                        );
                      })}
                    </div>
                    
                    {/* List of influencing patterns */}
                    <div className="mt-6 border-t border-slate-200 pt-4">
                      <h5 className="font-semibold text-sm text-slate-800 mb-3">Influencing Patterns:</h5>
                      <div className="flex flex-col gap-2">
                        {Array.from(new Set((explainResult.matches as any[]).map(m => m.pattern_id)))
                          .sort((a, b) => {
                            const freqA = explainResult.patternMap[a]?.frequency || 0;
                            const freqB = explainResult.patternMap[b]?.frequency || 0;
                            return freqB - freqA;
                          })
                          .map(pId => {
                          const pInfo = explainResult.patternMap[pId];
                          let seqStr = pInfo ? pInfo.sequence : "";
                          try {
                            const parsed = JSON.parse(seqStr);
                            if (Array.isArray(parsed)) seqStr = parsed.join(" → ");
                          } catch(e) {}
                          return (
                            <div key={pId} className="text-[11px] bg-white border border-slate-200 shadow-sm rounded-md p-2.5 text-slate-600">
                              <span className="font-bold text-slate-800">Pattern {pId}:</span> {seqStr} 
                              <span className="mx-2 text-slate-300">→</span> 
                              <span className="font-bold text-brand-600">{pInfo?.predicted_next}</span>
                              <span className="ml-3 font-mono text-slate-400 bg-slate-50 px-1.5 py-0.5 rounded">Freq: {pInfo?.frequency}</span>
                            </div>
                          );
                        })}
                        {explainResult.matches.length === 0 && (
                          <div className="text-xs text-slate-500 italic">No exact patterns matched this sequence.</div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div>
                    <h4 className="font-medium mb-3">{explainResult.method} Explanation</h4>
                    <div className="flex flex-col gap-4">
                      <img src={`${artifactUrl(runId, `${explainResult.path}/${explainResult.method.toLowerCase()}_summary.png`)}?t=${explainResult.timestamp}`} alt="Summary" className="border max-w-full" />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          )}
        </div>
      )}
    </div>
  );
}
