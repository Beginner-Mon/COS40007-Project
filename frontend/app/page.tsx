import PipelineFlow from "./components/PipelineFlow";

export default function Home() {
  return (
    <main className="min-h-screen bg-white dark:bg-slate-950 p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        <header>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white tracking-tight">🚀 ML Pipeline Manager</h1>
          <p className="text-slate-500 mt-2">
            Data Preprocessing & EDA workflow architecture powered by React Flow, Next.js, and TailwindCSS.
          </p>
        </header>

        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-slate-800 dark:text-slate-200">Processing Pipeline Overview</h2>
            <p className="text-sm text-slate-500 bg-slate-100 dark:bg-slate-800 px-3 py-1 rounded-full">
              Click a node to view code
            </p>
          </div>
          
          <div className="w-full h-full bg-white dark:bg-slate-900 p-2 rounded-xl shadow-lg border border-slate-200 dark:border-slate-800">
            <PipelineFlow />
          </div>
        </section>
      </div>
    </main>
  );
}
