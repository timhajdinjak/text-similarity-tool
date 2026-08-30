import { useState, useEffect } from "react";
import CorpusMode from "./CorpusMode";
import HomePage from "./components/HomePage";

export default function App() {
  const [page, setPage] = useState(() => window.location.hash === "#corpus" ? "corpus" : "home");

  useEffect(() => {
    const handler = () => setPage(window.location.hash === "#corpus" ? "corpus" : "home");
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);

  const goCorpus = () => { window.location.hash = "corpus"; setPage("corpus"); };
  const goHome = () => { window.location.hash = ""; setPage("home"); };

  if (page === "corpus") return <CorpusMode onBack={goHome} />;
  return <HomePage onGoCorpus={goCorpus} />;
}