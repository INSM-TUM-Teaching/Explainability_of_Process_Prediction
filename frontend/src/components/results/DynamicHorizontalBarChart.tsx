import React, { useState } from 'react';

export interface FeatureImportance {
  activity: string;
  importance: number;
}

export interface DynamicHorizontalBarChartProps {
  data: FeatureImportance[];
  baseValue?: number;
  task: string;
  method: string;
  title?: string;
  description?: React.ReactNode;
  isGlobal?: boolean;
}

export default function DynamicHorizontalBarChart({ data, baseValue, task, method, title, description, isGlobal }: DynamicHorizontalBarChartProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return <div className="text-gray-500 italic p-4 text-center">No explanation data available.</div>;
  }

  // The backend sorts ascending by absolute importance. 
  // We reverse it to show the most important features at the top.
  const sortedData = [...data].reverse();
  
  const maxPos = Math.max(...data.map(d => Math.max(0, d.importance)), 0.001);
  const maxNeg = Math.max(...data.map(d => Math.max(0, -d.importance)), 0.001);
  const totalSpan = maxPos + maxNeg;

  // We use 20% margin on both ends of the chart area for local explanations to allow space for values
  // Global explanations are strictly positive so we can start at 0 and use 95% width.
  const usableWidth = isGlobal ? 95 : 60; 
  const leftMargin = isGlobal ? 0 : 20;
  const zeroLinePct = leftMargin + (maxNeg / totalSpan) * usableWidth;

  const isTemporal = task === 'remaining_time' || task === 'time' || task === 'event_time';

  const formatValue = (val: number) => {
    if (isTemporal) {
      return `${val > 0 ? '+' : ''}${val.toFixed(2)} d`;
    }
    return `${val > 0 ? '+' : ''}${val.toFixed(3)}`;
  };

  return (
    <div className="w-full mt-4 pt-10 pb-4 bg-white border border-gray-100 rounded-lg shadow-sm px-6 relative">
      
      <h3 className={`absolute top-4 left-6 text-lg font-bold text-gray-800 ${title ? '' : 'capitalize'}`}>
        {title || `${method} Explanation`}
      </h3>
      {description && (
        <div className="absolute top-11 left-6 text-xs text-slate-500">
          {description}
        </div>
      )}

      <div 
        className={`grid w-full relative z-20 ${description ? 'mt-14' : 'mt-8'}`}
        style={{ gridTemplateColumns: 'max-content 1fr' }}
      >
        
        {/* Overlay for grid lines in the 2nd column */}
        <div 
          className="col-start-2 relative pointer-events-none z-0"
          style={{ gridRow: `1 / span ${sortedData.length}` }}
        >
            {/* Zero Line */}
            <div 
              className="absolute -top-4 bottom-0 w-px border-l border-dashed border-gray-400"
              style={{ left: `${zeroLinePct}%` }}
            />
            {baseValue !== undefined && (
              <div 
                className="absolute -top-10 z-30 px-3 py-1.5 bg-gray-50 border border-gray-200 rounded-md font-bold text-gray-800 shadow-sm text-xs whitespace-nowrap flex flex-col items-center pointer-events-auto"
                style={{ 
                  left: `${zeroLinePct}%`, 
                  transform: 'translateX(-50%)' 
                }}
              >
                Base Value: {isTemporal ? `${baseValue.toFixed(2)} d` : parseFloat(baseValue.toFixed(3))}
                <div className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 w-2.5 h-2.5 bg-gray-50 border-b border-r border-gray-200 rotate-45" />
              </div>
            )}
        </div>

        {sortedData.map((item, idx) => {
          const isZero = Math.abs(item.importance) < 1e-6;
          const isPositive = item.importance > 0;
          const barWidthPct = (Math.abs(item.importance) / totalSpan) * usableWidth;
          const isHovered = hoveredIdx === idx;
          const bgClass = isHovered ? "bg-slate-50/70" : "";
          const rowPos = idx + 1;
          
          return (
            <React.Fragment key={idx}>
              {/* Label Cell */}
              <div 
                className={`pr-3 flex justify-end items-center h-8 border-r-2 border-gray-800 z-10 transition-colors ${bgClass}`}
                style={{ gridRow: rowPos, gridColumn: 1 }}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
              >
                  <div className="flex flex-col items-end justify-center" title={item.activity}>
                    {item.activity.split('\n').map((part, i) => (
                      <span 
                        key={i} 
                        className={i === 0 ? "text-xs font-medium text-gray-700 whitespace-nowrap" : "text-[10px] text-gray-500 whitespace-nowrap"}
                      >
                        {part}
                      </span>
                    ))}
                  </div>
              </div>

              {/* Chart Cell */}
              <div 
                className={`relative h-8 flex items-center z-10 transition-colors ${bgClass}`}
                style={{ gridRow: rowPos, gridColumn: 2 }}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
              >
                
                {/* Zero Case */}
                {isZero && (
                  <div 
                    className="absolute h-5 flex items-center pl-2"
                    style={{ left: `${zeroLinePct}%` }}
                  >
                    <span className={`text-[11px] font-bold text-gray-400 transition-opacity whitespace-nowrap ${isHovered ? 'opacity-100' : 'opacity-0'}`}>
                      {formatValue(0)}
                    </span>
                  </div>
                )}

                {/* Negative Side */}
                {!isPositive && !isZero && (
                  <>
                    <div 
                      className="absolute h-5 bg-red-500 hover:bg-red-600 transition-colors duration-300 rounded-l-sm border border-red-700 shadow-sm"
                      style={{
                        right: `${100 - zeroLinePct}%`,
                        width: `${barWidthPct}%`,
                      }}
                    />
                    <div 
                      className="absolute h-5 flex items-center pr-2 pointer-events-none"
                      style={{
                        right: `${100 - zeroLinePct + barWidthPct}%`,
                      }}
                    >
                      <span className={`text-[11px] font-bold text-red-700 transition-opacity whitespace-nowrap ${isHovered ? 'opacity-100' : 'opacity-0'}`}>
                        {formatValue(item.importance)}
                      </span>
                    </div>
                  </>
                )}

                {/* Positive Side */}
                {isPositive && !isZero && (
                  <>
                    <div 
                      className={`absolute h-5 transition-all duration-300 rounded-r-sm border shadow-sm ${isGlobal ? 'bg-[#2855A3] hover:opacity-80 border-[#2855A3]' : 'bg-green-500 hover:bg-green-600 border-green-700'}`}
                      style={{
                        left: `${zeroLinePct}%`,
                        width: `${barWidthPct}%`,
                      }}
                    />
                    <div 
                      className="absolute h-5 flex items-center pl-2 pointer-events-none"
                      style={{
                        left: `${zeroLinePct + barWidthPct}%`,
                      }}
                    >
                      <span className={`text-[11px] font-bold transition-opacity whitespace-nowrap ${isGlobal ? 'text-[#2855A3]' : 'text-green-700'} ${isHovered ? 'opacity-100' : 'opacity-0'}`}>
                        {formatValue(item.importance)}
                      </span>
                    </div>
                  </>
                )}

              </div>
            </React.Fragment>
          );
        })}
      </div>

      {!isGlobal && (
        <div className="mt-6 flex justify-center gap-8 border-t border-gray-100 pt-4 mx-6">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 border border-green-700 rounded-sm" />
            <span className="text-xs text-gray-600 font-medium">
              {isTemporal ? 'Increases Duration' : 'Supports'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 border border-red-700 rounded-sm" />
            <span className="text-xs text-gray-600 font-medium">
              {isTemporal ? 'Decreases Duration' : 'Contradicts'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
