import { NavLink } from "react-router-dom";
import { BrainCircuit, ChartSpline, Microscope, Search, Workflow } from "lucide-react";

const links = [
  { to: "/", label: "Overview", icon: BrainCircuit },
  { to: "/demo", label: "Live Demo", icon: Search },
  { to: "/experiments", label: "Experiments", icon: ChartSpline },
  { to: "/analysis", label: "Analysis", icon: Microscope },
  { to: "/architecture", label: "Architecture", icon: Workflow },
];

export default function Navbar() {
  return (
    <header className="topbar">
      <div className="brand-block">
        <div className="brand-mark">TS</div>
        <div>
          <p className="brand-title">two-stage ranking studio</p>
          <p className="brand-subtitle">retrieval + reranking dashboard</p>
        </div>
      </div>
      <nav className="nav-links">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Icon size={16} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
