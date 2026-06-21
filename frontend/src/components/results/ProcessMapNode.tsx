import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

const ProcessMapNode = ({ data }: any) => {
  return (
    <div className="px-5 py-4 shadow-md rounded-2xl bg-brand-600 border border-brand-700 min-w-[220px] transition-all hover:shadow-lg hover:bg-brand-700 group">
      {/* Connector dots */}
      <Handle 
        type="target" 
        position={Position.Top} 
        className="w-2.5 h-2.5 bg-brand-300 border-2 border-brand-600 !-top-1.5" 
      />
      
      <div className="flex flex-col gap-1.5 text-center text-white">
        <div className="font-bold text-[14px] leading-tight px-2">
          {data.label}
        </div>
        
        <div className="flex items-center justify-center gap-2 mt-1">
          <div className="px-2 py-0.5 rounded bg-white/20 text-white text-[10px] font-bold uppercase tracking-wider border border-white/30">
            {data.count.toLocaleString()} 
          </div>
          <span className="text-white/60 text-[10px] font-medium uppercase tracking-tight">Total Events</span>
        </div>
      </div>

      <Handle 
        type="source" 
        position={Position.Bottom} 
        className="w-2.5 h-2.5 bg-brand-300 border-2 border-brand-600 !-bottom-1.5" 
      />
    </div>
  );
};

export default memo(ProcessMapNode);
