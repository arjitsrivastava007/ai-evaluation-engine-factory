import { Link, Route, Routes } from "react-router-dom";
import { Home } from "./pages/Home";
import { Workspace } from "./pages/Workspace";

export function App() {
  return (
    <div className="shell">
      <header className="topbar">
        <Link className="brand" to="/">
          <div className="mark">Ee</div>
          <div>
            <strong>Evaluation Factory</strong>
            <span>RAG · chatbot · agent</span>
          </div>
        </Link>
        <nav>
          <Link to="/">New evaluation</Link>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/runs/:runId" element={<Workspace />} />
      </Routes>
    </div>
  );
}
