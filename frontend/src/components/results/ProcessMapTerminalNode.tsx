import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Play, Square } from 'lucide-react';

const ProcessMapTerminalNode = ({ data }: any) => {
  const isStart = data.type === 'start';
  
  return (
    <div className={`flex items-center justify-center w-12 h-12 rounded-full shadow-lg border-[3px] transition-all hover:scale-110 ${isStart ? 'bg-emerald-50 border-emerald-500 text-emerald-600' : 'bg-rose-50 border-rose-500 text-rose-600'}`}>
      {!isStart && <Handle type="target" position={Position.Top} className="w-2 h-2 opacity-0" />}
      <div className="flex flex-col items-center justify-center">
        {isStart ? <Play className="fill-current" size={24} /> : <Square className="fill-current" size={20} />}
      </div>
      {isStart && <Handle type="source" position={Position.Bottom} className="w-2 h-2 opacity-0" />}
    </div>
  );
};

export default memo(ProcessMapTerminalNode);
