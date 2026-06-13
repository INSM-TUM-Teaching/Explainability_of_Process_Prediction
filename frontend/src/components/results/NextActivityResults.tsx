import { useEffect, useState } from "react";
import { artifactUrl, artifactsZipUrl } from "../../lib/api";
import { explainOnDemand } from "../../lib/api_explain";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import GlobalResults from "./GlobalResults";

type PredictionRecord = {
  case_id: string;
  case_index: number;
  sequence: string;
  true_next_activity: string;
  predicted_next_activity: string;
  confidence_percent: number;
};

export default function NextActivityResults({ runId, summary, onBackToPipeline }: any) {
  const [predictions, setPredictions] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  
  useEffect(() => {
    async function loadData() {
      const predFile = summary.request.model_type === "transformer" ? "transformer_predictions.json" : "gnn_predictions.json";
      try {
        const res = await fetch(artifactUrl(runId, predFile));
        if (res.ok) {
          const data = await res.json();
          setPredictions(data);
        }
      } catch (err) {
        console.error("Failed to load predictions", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [runId, summary]);

  const caseIds = Array.from(new Set(predictions.map(p => p.case_id)));
  
  const filteredCases = caseIds.filter(cid => {
    const caseRecs = predictions.filter(p => p.case_id === cid);
    const vId = String(caseRecs[0]?.variant_id || "");
    const searchTerm = search.toLowerCase();
    return cid.toLowerCase().includes(searchTerm) || vId.toLowerCase().includes(searchTerm);
  });

  const config = summary.request.config || {};
  const metrics = summary.metrics || {};
  
  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Top Section */}
      <div className="flex flex-col gap-4 rounded-2xl border border-brand-100 bg-white p-6 shadow-sm md:flex-row md:items-start md:justify-between">
        <div>
          <div className="text-sm font-medium uppercase tracking-[0.18em] text-brand-500">
            Next Activity Prediction Results
          </div>
          <h1 className="mt-2 text-2xl font-semibold text-brand-900">Run {runId}</h1>
          <p className="mt-2 max-w-3xl text-sm text-slate-600">
            Uploaded file: <strong>{summary.dataset?.filename}</strong> | Model: <strong>{summary.request?.model_type}</strong>
          </p>
        </div>
        <div className="flex gap-3">
          <button onClick={onBackToPipeline} className="rounded-lg border px-4 py-2 text-sm font-medium hover:bg-slate-50">Back</button>
          <a href={artifactsZipUrl(runId)} className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700">
            Download all artifacts
          </a>
        </div>
      </div>

      <Tabs defaultValue="global" className="w-full mt-2">
        <TabsList className="mb-4">
          <TabsTrigger value="global">Global Overview</TabsTrigger>
          <TabsTrigger value="local">Local Case Analysis</TabsTrigger>
        </TabsList>
        <TabsContent value="global">
          <GlobalResults runId={runId} datasetId={summary.dataset?.dataset_id} summary={summary} />
        </TabsContent>
        <TabsContent value="local">
          <div className="rounded-2xl border border-brand-100 bg-white p-6 shadow-sm mt-4">
            <div className="flex items-center justify-between gap-3 mb-6">
              <h2 className="text-lg font-semibold text-brand-900">Local Prediction Data</h2>
              <input 
                type="text" 
                placeholder="Search by Case ID or Variant ID..." 
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="border rounded px-3 py-1 text-sm w-72"
              />
            </div>

            {loading ? (
              <div>Loading prediction data...</div>
            ) : (
              <div className="flex flex-col gap-4">
                {filteredCases.map(cid => (
                  <CasePredictionBlock 
                    key={cid} 
                    caseId={cid} 
                    records={predictions.filter(p => p.case_id === cid)} 
                    runId={runId}
                    modelType={summary.request?.model_type}
                  />
                ))}
                {filteredCases.length === 0 && <div>No cases found matching search.</div>}
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-brand-100 bg-white p-5 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wide text-brand-500">{label}</div>
      <div className="mt-2 text-base font-semibold text-brand-900 break-words">{value}</div>
    </div>
  );
}

function CasePredictionBlock({ caseId, records, runId, modelType }: { caseId: string, records: any[], runId: string, modelType: string }) {
  const [expanded, setExpanded] = useState(false);
  const maxIndex = Math.max(...records.map(r => r.case_index));
  const [selectedIndex, setSelectedIndex] = useState(maxIndex);

  const selectedRecord = records.find(r => r.case_index === selectedIndex) || records[0];
  const variantId = selectedRecord?.variant_id;

  const explains = modelType === "transformer" ? ["SHAP", "LIME"] : ["Gradient", "GraphLIME"];
  const [selectedExplain, setSelectedExplain] = useState(explains[0]);
  const [explainResult, setExplainResult] = useState<any>(null);
  const [explaining, setExplaining] = useState(false);

  // When expanding or changing index, clear old explanation
  useEffect(() => {
    setExplainResult(null);
  }, [expanded, selectedIndex]);

  const handleExplain = async () => {
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
  };

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
              {selectedRecord.sequence.split(',').map((act, i, arr) => (
                <div key={i} className="flex items-center gap-2">
                  <div className="bg-brand-50 border border-brand-500 text-brand-800 px-3 py-1.5 rounded-md shadow-sm text-xs font-medium whitespace-nowrap">
                    {act.trim()}
                  </div>
                  {i < arr.length - 1 && (
                    <svg className="w-4 h-4 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                  )}
                </div>
              ))}
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4">
            <div className="p-3 border rounded">
              <div className="text-slate-500 mb-1">True Next Activity</div>
              <div className="font-semibold">{selectedRecord.true_next_activity}</div>
            </div>
            <div className={`p-3 border rounded ${isCorrect ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
              <div className={`mb-1 ${isCorrect ? 'text-green-700' : 'text-red-700'}`}>Predicted Next Activity</div>
              <div className={`font-semibold ${isCorrect ? 'text-green-800' : 'text-red-800'}`}>{selectedRecord.predicted_next_activity}</div>
            </div>
            <div className="p-3 border rounded">
              <div className="text-slate-500 mb-1">Confidence</div>
              <div className="font-semibold">{Number(selectedRecord.confidence_percent).toFixed(2)}%</div>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t flex flex-col gap-3">
            <div className="flex gap-3 items-center">
              <strong>Generate Explanation:</strong>
              <select className="border rounded px-2 py-1" value={selectedExplain} onChange={e => setSelectedExplain(e.target.value)}>
                {explains.map(ex => <option key={ex} value={ex}>{ex}</option>)}
              </select>
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
                <h4 className="font-medium mb-3">{explainResult.method} Explanation</h4>
                <div className="flex flex-col gap-4">
                  <img src={`${artifactUrl(runId, `${explainResult.path}/${explainResult.method.toLowerCase()}_summary.png`)}?t=${explainResult.timestamp}`} alt="Summary" className="border max-w-full" />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
