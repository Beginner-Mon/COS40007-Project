"use client";

import { useState, useCallback } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  BackgroundVariant,
  NodeMouseHandler
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Terminal, Code, X } from 'lucide-react';

const initialNodes: Node[] = [
  { id: '1', type: 'input', position: { x: 250, y: 0 }, data: { label: '📂 Load Raw Output Data' } },
  { id: '2', position: { x: 250, y: 100 }, data: { label: '🔗 Merge Velocity & Accel Sensors' } },
  { id: '3', position: { x: 250, y: 200 }, data: { label: '🏷️ Map & Filter Target Labels' } },
  { id: '4', position: { x: 250, y: 300 }, data: { label: '🎯 Derive Target Subclasses' } },
  { id: '5', position: { x: 250, y: 400 }, data: { label: '⚙️ Engineer Features' } },
  { id: '6', position: { x: 250, y: 500 }, data: { label: '✂️ Generate Sequence Windows' } },
  { id: '7', position: { x: 250, y: 600 }, data: { label: '📏 Padding Normalization (Len=60)' } },
  { id: '8', position: { x: 250, y: 700 }, data: { label: '🧹 Clean Final Features' } },
  { id: '9', position: { x: 100, y: 850 }, data: { label: '🔄 K-Fold Split Strategy' } },
  { id: '10', position: { x: 400, y: 850 }, data: { label: '📊 Holdout Split Strategy' } },
  { id: '11', type: 'output', position: { x: 250, y: 1000 }, data: { label: '🧠 Execute Model Training' }, style: { backgroundColor: '#10b981', color: 'white' } },
];

const initialEdges: Edge[] = [
  { id: 'e1-2', source: '1', target: '2', animated: true },
  { id: 'e2-3', source: '2', target: '3', animated: true },
  { id: 'e3-4', source: '3', target: '4', animated: true },
  { id: 'e4-5', source: '4', target: '5', animated: true },
  { id: 'e5-6', source: '5', target: '6', animated: true },
  { id: 'e6-7', source: '6', target: '7', animated: true },
  { id: 'e7-8', source: '7', target: '8', animated: true },
  { id: 'e8-9', source: '8', target: '9', animated: true },
  { id: 'e8-10', source: '8', target: '10', animated: true },
  { id: 'e9-11', source: '9', target: '11', animated: true },
  { id: 'e10-11', source: '10', target: '11', animated: true },
];

const nodeSnippets: Record<string, string> = {
  '1': 'print("[phase=data] Loading CSV files...")\ndfs = []\nfor pid in ["P1", "P2"]:\n    for act in ["boning", "slicing"]:\n        csv_path = DATA_DIR / f"{pid}_{act}.csv"\n        dfs.append(pd.read_csv(csv_path))\nraw_df = pd.concat(dfs, ignore_index=True)',
  '2': '# Combines Velocity & Acceleration schemas\nmerged_df, base_feature_cols = merge_velocity_and_acceleration(raw_df)',
  '3': 'label_map = dict(cfg.task.label_mapping)\nmerged_df["Label"] = merged_df["Label"].replace(label_map)\nmerged_df.dropna(subset=["Label"], inplace=True)\nmerged_df["Label"] = merged_df["Label"].astype("int64")',
  '4': '# Automatically derive subclass targets based on cfg\nif TARGET_COL == "sharpness_class":\n    merged_df = derive_sharpness_class(merged_df)',
  '5': '# Generate custom engineered columns beyond base kinematics\nmerged_df, feature_cols = engineer_features(merged_df, base_feature_cols)',
  '6': '# Slice continuous data into overlapping 3D tensors\nX_windows, y_windows, window_meta_df = create_windows(\n    merged_df, feature_cols, cfg.data.window_size, cfg.data.stride, TARGET_COL\n)',
  '7': '# Constant width vector lengths\nX_all = pad_windows_to_60(X_windows, target_len=60)',
  '8': '# Drop sparse or NaN feature derivations\nX_all = clean_features(X_all)\ny_all = label_encoder.fit_transform(y_windows).astype(np.int64)',
  '9': 'print("Initializing K-Fold...")\nrun_kfold_training(X_all, y_all, window_meta_df, cfg, run_dir, label_encoder, device)',
  '10': 'print("Initializing Holdout...")\nrun_holdout_training(X_all, y_all, window_meta_df, cfg, run_dir, label_encoder, device)',
  '11': '# model.tcn, model.gru, model.bilstm execution\n# Weights & biases tracked via active MLflow session'
};

export default function PipelineFlow() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [activeSnippet, setActiveSnippet] = useState<{title: string, code: string} | null>(null);

  const onNodeClick: NodeMouseHandler = useCallback((event, node) => {
    setActiveSnippet({
      title: node.data.label as string,
      code: nodeSnippets[node.id] || '# No snippet mapped.'
    });
  }, []);

  return (
    <div className="relative w-full h-full flex bg-slate-50 dark:bg-slate-900 rounded-lg overflow-hidden border border-slate-200 shadow-sm">
      <div className="flex-grow h-[80vh] min-h-[600px]">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          fitView
          fitViewOptions={{ padding: 0.2 }}
        >
          <Controls className="bg-white shadow-md rounded-md" />
          <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
        </ReactFlow>
      </div>

      {activeSnippet && (
        <div className="absolute top-4 right-4 w-96 max-h-[80%] bg-slate-900 shadow-2xl rounded-xl z-50 flex flex-col font-mono text-sm border border-slate-700 animate-in fade-in slide-in-from-right-4 duration-200">
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900/50 rounded-t-xl">
            <div className="flex items-center gap-2 text-slate-200 font-semibold truncate">
              <Terminal size={16} className="text-emerald-400" />
              <span className="truncate">{activeSnippet.title}</span>
            </div>
            <button 
              onClick={() => setActiveSnippet(null)}
              className="text-slate-400 hover:text-white transition-colors"
            >
              <X size={18} />
            </button>
          </div>
          <div className="p-4 overflow-auto overflow-x-auto whitespace-pre">
            <code className="text-emerald-300">
                {activeSnippet.code.split('\n').map((line, i) => (
                    <div key={i} className="table-row">
                        <span className="table-cell text-slate-600 select-none pr-4 text-right sm:inline-block w-6">{i + 1}</span>
                        <span className="table-cell">{line}</span>
                    </div>
                ))}
            </code>
          </div>
        </div>
      )}
    </div>
  );
}
