import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

const ProcessMapNode = ({ data }: any) => {
  return (
    <div className="px-6 py-3 shadow-lg rounded-xl bg-gradient-to-b from-brand-500 to-brand-700 border-2 border-brand-800 text-white min-w-[180px] text-center transition-all hover:scale-105">
      <Handle type="target" position={Position.Top} className="w-3 h-3 bg-brand-200 border-2 border-brand-800" />
      <div className="flex flex-col gap-1">
        <div className="font-bold text-sm leading-tight">{data.label}</div>
        <div className="h-px bg-white/20 w-full my-1" />
        <div className="text-[11px] font-mono opacity-90">{data.count.toLocaleString()}</div>
      </div>
      <Handle type="source" position={Position.Bottom} className="w-3 h-3 bg-brand-200 border-2 border-brand-800" />
    </div>
  );
};

export default memo(ProcessMapNode);
