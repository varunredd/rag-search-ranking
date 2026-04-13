import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import DemoPage from "./pages/DemoPage";
import ExperimentsPage from "./pages/ExperimentsPage";
import AnalysisPage from "./pages/AnalysisPage";
import ArchitecturePage from "./pages/ArchitecturePage";

export default function App() {
  return (
    <div className="app-shell">
      <Navbar />
      <main className="page-shell">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/demo" element={<DemoPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/architecture" element={<ArchitecturePage />} />
        </Routes>
      </main>
    </div>
  );
}
