import { useState } from "react";
import { AnalyzeTab } from "./components/AnalyzeTab";
import { FeedbackTab } from "./components/FeedbackTab";
import { ModelsTab } from "./components/ModelsTab";
import { ToastProvider } from "./components/Toast";
import { TopBar, type Tab } from "./components/TopBar";
import { useHealth } from "./lib/useHealth";

export default function App() {
  const [tab, setTab] = useState<Tab>("analyze");
  const { health, error, refresh } = useHealth();

  return (
    <ToastProvider>
      <div className="min-h-full">
        <TopBar tab={tab} onTab={setTab} health={health} healthError={error} />
        <main className="mx-auto max-w-7xl px-4 py-6 md:px-6">
          {/* Tabs stay mounted so editor contents, results and a running retrain survive tab switches. */}
          <div hidden={tab !== "analyze"} role="tabpanel">
            <AnalyzeTab health={health} active={tab === "analyze"} onAnalyzed={refresh} />
          </div>
          <div hidden={tab !== "feedback"} role="tabpanel">
            <FeedbackTab active={tab === "feedback"} onRetrained={refresh} />
          </div>
          <div hidden={tab !== "models"} role="tabpanel">
            <ModelsTab health={health} />
          </div>
        </main>
      </div>
    </ToastProvider>
  );
}
