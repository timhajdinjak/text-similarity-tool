import { useState, useEffect, useCallback, useRef } from "react";

const API_URL = "http://localhost:5000";

const TYPE_COLORS = {
    "Token / Set-Based": { bg: "rgba(99,102,241,0.15)", border: "#6366f1", accent: "#818cf8" },
    "Vector / TF-IDF": { bg: "rgba(16,185,129,0.15)", border: "#10b981", accent: "#34d399" },
    "String / Character-Based": { bg: "rgba(245,158,11,0.15)", border: "#f59e0b", accent: "#fbbf24" },
    "Embedding / Semantic": { bg: "rgba(236,72,153,0.15)", border: "#ec4899", accent: "#f472b6" },
    "Token / N-Gram": { bg: "rgba(59,130,246,0.15)", border: "#3b82f6", accent: "#60a5fa" },
    "Sequence-Based": { bg: "rgba(168,85,247,0.15)", border: "#a855f7", accent: "#c084fc" },
    "NLG Evaluation": { bg: "rgba(20,184,166,0.15)", border: "#14b8a6", accent: "#2dd4bf" },
    "Information-Theoretic": { bg: "rgba(239,68,68,0.15)", border: "#ef4444", accent: "#f87171" },
};

const METHOD_TYPES = {
    jaccard: "Token / Set-Based", dice: "Token / Set-Based",
    overlap: "Token / Set-Based", containment: "Token / Set-Based",
    tfidf_cosine: "Vector / TF-IDF",
    levenshtein: "String / Character-Based", jaro_winkler: "String / Character-Based",
    trigram: "String / Character-Based",
    bigram_overlap: "Token / N-Gram",
    lcs: "Sequence-Based", hamming: "Sequence-Based",
    bleu: "NLG Evaluation", rouge_l: "NLG Evaluation",
    rouge_2: "NLG Evaluation", meteor: "NLG Evaluation",
    kl_divergence: "Information-Theoretic", jensen_shannon: "Information-Theoretic",
    ncd: "Information-Theoretic",
    embedding_cosine: "Embedding / Semantic", wmd: "Embedding / Semantic",
    soft_cosine: "Embedding / Semantic", bert_score: "Embedding / Semantic",
};

const METHOD_NAMES = {
    jaccard: "Jaccard", dice: "Sørensen–Dice", overlap: "Overlap Coeff.",
    containment: "Containment", tfidf_cosine: "TF-IDF Cosine",
    levenshtein: "Levenshtein", jaro_winkler: "Jaro-Winkler",
    trigram: "Char Trigram", bigram_overlap: "Bigram Overlap",
    lcs: "LCS", hamming: "Hamming",
    bleu: "BLEU", rouge_l: "ROUGE-L", rouge_2: "ROUGE-2", meteor: "METEOR",
    kl_divergence: "KL Divergence", jensen_shannon: "Jensen-Shannon", ncd: "NCD",
    embedding_cosine: "Embedding Cosine", wmd: "WMD",
    soft_cosine: "Soft Cosine", bert_score: "BERTScore",
};

const ALL_METHODS = Object.keys(METHOD_NAMES);

const EXAMPLE_QUERY = "machine learning neural networks deep learning";
const EXAMPLE_DOCS = [
    "Deep learning and neural networks have revolutionized artificial intelligence research.",
    "The stock market experienced significant volatility due to geopolitical tensions.",
    "Convolutional neural networks achieve state-of-the-art results in image recognition tasks.",
    "Recurrent neural networks and LSTMs are used for natural language processing.",
    "The weather forecast predicts heavy rainfall across the eastern coast this weekend.",
    "Gradient descent optimization is fundamental to training machine learning models.",
    "A new restaurant opened downtown serving authentic Italian cuisine and wood-fired pizza.",
    "Transfer learning allows pre-trained models to be fine-tuned on new tasks efficiently.",
    "The championship game ended in a dramatic overtime victory for the home team.",
    "Backpropagation through time enables training of recurrent neural architectures.",
];

function ModelSelector({ groups, selectedId, loadedId, onChange }) {
    const [open, setOpen] = useState(false);
    const ref = useRef(null);

    useEffect(() => {
        const h = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
        document.addEventListener("mousedown", h);
        return () => document.removeEventListener("mousedown", h);
    }, []);

    const selectedModel = groups.flatMap(g => g.models).find(m => m.id === selectedId);
    const isLoaded = selectedId === loadedId;

    return (
        <div ref={ref} style={{ position: "relative" }}>
            <div style={{ fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color: "#6366f1", letterSpacing: "0.1em", marginBottom: "8px", display: "flex", alignItems: "center", gap: "8px" }}>
                EMBEDDING MODEL
                {loadedId && (
                    <span style={{ background: isLoaded ? "rgba(16,185,129,0.15)" : "rgba(245,158,11,0.15)", border: `1px solid ${isLoaded ? "#10b981" : "#f59e0b"}`, color: isLoaded ? "#34d399" : "#fbbf24", borderRadius: "4px", padding: "1px 7px", fontSize: "10px" }}>
                        {isLoaded ? "✓ loaded" : "⟳ will swap"}
                    </span>
                )}
            </div>
            <button onClick={() => setOpen(o => !o)}
                style={{ width: "100%", background: "#0f172a", border: "1px solid #1e293b", borderRadius: "12px", padding: "12px 16px", color: "#e2e8f0", fontFamily: "'DM Sans', sans-serif", fontSize: "14px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", textAlign: "left" }}>
                <div>
                    <div style={{ fontWeight: 500 }}>{selectedModel?.label ?? selectedId}</div>
                    {selectedModel?.note && <div style={{ fontSize: "11px", color: "#475569", marginTop: "2px", fontFamily: "JetBrains Mono, monospace" }}>{selectedModel.note}</div>}
                </div>
                <span style={{ color: "#475569", transform: open ? "rotate(180deg)" : "none", transition: "transform 0.2s", marginLeft: "12px" }}>▼</span>
            </button>
            {open && (
                <div style={{ position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0, background: "#0d1526", border: "1px solid #1e293b", borderRadius: "12px", zIndex: 300, maxHeight: "340px", overflowY: "auto", boxShadow: "0 8px 40px rgba(0,0,0,0.6)" }}>
                    {groups.map(group => (
                        <div key={group.group}>
                            <div style={{ padding: "10px 16px 6px", fontSize: "10px", fontFamily: "JetBrains Mono, monospace", color: "#6366f1", letterSpacing: "0.1em", borderBottom: "1px solid #1e293b", background: "#080f1a", position: "sticky", top: 0 }}>
                                {group.group.toUpperCase()}
                            </div>
                            {group.models.map(model => {
                                const isActive = model.id === selectedId;
                                const isCurrentlyLoaded = model.id === loadedId;
                                return (
                                    <button key={model.id} onClick={() => { onChange(model.id); setOpen(false); }}
                                        style={{ width: "100%", background: isActive ? "rgba(99,102,241,0.1)" : "transparent", border: "none", padding: "10px 16px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", textAlign: "left" }}>
                                        <div>
                                            <div style={{ color: isActive ? "#818cf8" : "#94a3b8", fontSize: "13px", fontWeight: isActive ? 600 : 400 }}>{model.label}</div>
                                            <div style={{ color: "#374151", fontSize: "11px", fontFamily: "JetBrains Mono, monospace", marginTop: "2px" }}>{model.note}</div>
                                        </div>
                                        {isCurrentlyLoaded && <span style={{ fontSize: "10px", background: "rgba(16,185,129,0.15)", border: "1px solid #10b981", color: "#34d399", borderRadius: "4px", padding: "1px 6px" }}>in RAM</span>}
                                    </button>
                                );
                            })}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

function RankBadge({ rank, total }) {
    const pct = rank / total;
    const color = pct <= 0.2 ? "#34d399" : pct <= 0.5 ? "#fbbf24" : "#f87171";
    return (
        <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: "22px", height: "22px", borderRadius: "50%", background: `${color}20`, border: `1px solid ${color}60`, fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color, fontWeight: 700, flexShrink: 0 }}>
            {rank}
        </span>
    );
}

function ScoreBar({ score, color }) {
    return (
        <div style={{ display: "flex", alignItems: "center", gap: "8px", minWidth: 0 }}>
            <div style={{ flex: 1, height: "4px", background: "#0f172a", borderRadius: "2px", overflow: "hidden" }}>
                <div style={{ width: `${Math.round(score * 100)}%`, height: "100%", background: color, borderRadius: "2px", transition: "width 0.5s cubic-bezier(.4,0,.2,1)" }} />
            </div>
            <span style={{ fontSize: "12px", fontFamily: "JetBrains Mono, monospace", color, fontWeight: 600, flexShrink: 0, minWidth: "38px", textAlign: "right" }}>
                {(score * 100).toFixed(1)}%
            </span>
        </div>
    );
}

function DivergenceTable({ rankings, docLabels, selectedMethods }) {
    if (!rankings || selectedMethods.length === 0) return null;
    const nDocs = docLabels.length;

    const rankMatrix = docLabels.map((_, di) =>
        selectedMethods.map(mkey => {
            const r = rankings[mkey]?.find(e => e.doc_index === di);
            return r ? r.rank : nDocs;
        })
    );

    const rowVariances = rankMatrix.map(row => {
        const mean = row.reduce((a, b) => a + b, 0) / row.length;
        const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
        return Math.sqrt(variance);
    });

    const rankColor = (rank, total) => {
        const pct = (rank - 1) / (total - 1 || 1);
        if (pct < 0.33) return `rgba(52,211,153,${0.15 + (1 - pct / 0.33) * 0.45})`;
        if (pct < 0.66) return `rgba(251,191,36,${0.15 + 0.3})`;
        return `rgba(248,113,113,${0.15 + pct * 0.45})`;
    };

    const maxVariance = Math.max(...rowVariances, 1);

    return (
        <div style={{ overflowX: "auto", borderRadius: "14px", border: "1px solid #1e293b" }}>
            <table style={{ borderCollapse: "collapse", width: "100%", minWidth: `${200 + selectedMethods.length * 70}px` }}>
                <thead>
                    <tr style={{ background: "#080f1a" }}>
                        <th style={{ padding: "10px 14px", textAlign: "left", fontSize: "10px", fontFamily: "JetBrains Mono, monospace", color: "#475569", letterSpacing: "0.08em", whiteSpace: "nowrap", position: "sticky", left: 0, background: "#080f1a", zIndex: 2, minWidth: "180px" }}>
                            DOCUMENT
                        </th>
                        <th style={{ padding: "10px 8px", fontSize: "9px", fontFamily: "JetBrains Mono, monospace", color: "#374151", letterSpacing: "0.06em", textAlign: "center", minWidth: "52px", whiteSpace: "nowrap" }}>
                            SPREAD
                        </th>
                        {selectedMethods.map(mkey => {
                            const c = TYPE_COLORS[METHOD_TYPES[mkey]] || { accent: "#818cf8", border: "#6366f1" };
                            return (
                                <th key={mkey} style={{ padding: "10px 6px", fontSize: "9px", fontFamily: "JetBrains Mono, monospace", color: c.accent, letterSpacing: "0.05em", textAlign: "center", minWidth: "60px", whiteSpace: "nowrap", borderLeft: "1px solid #0f172a" }}>
                                    {METHOD_NAMES[mkey]}
                                </th>
                            );
                        })}
                    </tr>
                </thead>
                <tbody>
                    {rankMatrix.map((row, di) => {
                        const spread = rowVariances[di];
                        const spreadPct = spread / maxVariance;
                        return (
                            <tr key={di} style={{ borderTop: "1px solid #0f172a" }}>
                                <td style={{ padding: "9px 14px", fontSize: "12px", color: "#94a3b8", fontFamily: "'DM Sans', sans-serif", position: "sticky", left: 0, background: "#07111f", zIndex: 1, maxWidth: "220px" }}>
                                    <span style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={docLabels[di]}>
                                        {docLabels[di]}
                                    </span>
                                </td>
                                <td style={{ padding: "9px 8px", textAlign: "center" }}>
                                    <div title={`Rank std dev: ${spread.toFixed(1)}`}
                                        style={{ width: "36px", height: "6px", borderRadius: "3px", margin: "0 auto", background: `rgba(251,191,36,${0.15 + spreadPct * 0.7})`, border: `1px solid rgba(251,191,36,${0.2 + spreadPct * 0.5})` }} />
                                </td>
                                {row.map((rank, mi) => (
                                    <td key={mi} style={{ padding: "8px 6px", textAlign: "center", background: rankColor(rank, nDocs), borderLeft: "1px solid #0a1628", transition: "background 0.3s" }}>
                                        <span style={{ fontSize: "12px", fontFamily: "JetBrains Mono, monospace", color: rank <= Math.ceil(nDocs / 3) ? "#34d399" : rank <= Math.ceil(nDocs * 2 / 3) ? "#fbbf24" : "#f87171", fontWeight: 600 }}>
                                            #{rank}
                                        </span>
                                    </td>
                                ))}
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default function CorpusMode({ onBack }) {
    const [query, setQuery] = useState("");
    const [docs, setDocs] = useState(Array(10).fill(""));
    const [nDocs, setNDocs] = useState(10);

    const [modelGroups, setModelGroups] = useState([]);
    const [selectedModel, setSelectedModel] = useState("all-MiniLM-L6-v2");
    const [loadedModelId, setLoadedModelId] = useState(null);
    const [spacyLanguages, setSpacyLanguages] = useState([]);
    const [ppLowercase, setPpLowercase] = useState(true);
    const [ppPunct, setPpPunct] = useState(true);
    const [ppStopwords, setPpStopwords] = useState(false);
    const [ppLang, setPpLang] = useState("en_core_web_sm");

    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const [viewMode, setViewMode] = useState("ranked");
    const [selectedMethod, setSelectedMethod] = useState("embedding_cosine");
    const [activeMethods, setActiveMethods] = useState(ALL_METHODS);
    const [highlightDoc, setHighlightDoc] = useState(null);

    useEffect(() => {
        fetch(`${API_URL}/models`).then(r => r.json()).then(d => {
            setModelGroups(d.groups || []);
            if (d.loaded_model) setLoadedModelId(d.loaded_model);
            if (d.default_model) setSelectedModel(d.default_model);
        }).catch(() => { });
        fetch(`${API_URL}/spacy-models`).then(r => r.json()).then(d => {
            setSpacyLanguages(d.languages || []);
        }).catch(() => { });
    }, []);

    const handleDocChange = (i, val) => {
        setDocs(prev => { const n = [...prev]; n[i] = val; return n; });
    };

    const loadExample = () => {
        setQuery(EXAMPLE_QUERY);
        setDocs(EXAMPLE_DOCS.map((d, i) => i < nDocs ? d : ""));
    };

    const handleNDocsChange = (n) => {
        setNDocs(n);
        setDocs(prev => {
            const next = [...prev];
            while (next.length < n) next.push("");
            return next.slice(0, n);
        });
    };

    const toggleMethod = (mkey) => {
        setActiveMethods(prev =>
            prev.includes(mkey)
                ? prev.length > 1 ? prev.filter(k => k !== mkey) : prev
                : [...prev, mkey]
        );
    };

    const retrieve = useCallback(async () => {
        const filledDocs = docs.slice(0, nDocs).filter(d => d.trim());
        if (!query.trim()) { setError("Please enter a query."); return; }
        if (filledDocs.length < 2) { setError("Please enter at least 2 documents."); return; }

        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const res = await fetch(`${API_URL}/corpus`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query,
                    documents: docs.slice(0, nDocs).map(d => d.trim() || "(empty)"),
                    model_id: selectedModel,
                    lowercase: ppLowercase,
                    remove_punct: ppPunct,
                    remove_stopwords: ppStopwords,
                    spacy_model: ppLang,
                }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
            setResults(data);
            setLoadedModelId(data.model_id);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    }, [query, docs, nDocs, selectedModel, ppLowercase, ppPunct, ppStopwords, ppLang]);

    const methodGroups = Object.entries(
        ALL_METHODS.reduce((acc, k) => {
            const t = METHOD_TYPES[k];
            if (!acc[t]) acc[t] = [];
            acc[t].push(k);
            return acc;
        }, {})
    );

    const docLabels = results
        ? results.docs_preprocessed.map((d, i) =>
            docs[i]?.trim().slice(0, 55) + (docs[i]?.trim().length > 55 ? "…" : "") || `Document ${i + 1}`
        )
        : [];

    const selectedColors = TYPE_COLORS[METHOD_TYPES[selectedMethod]] || { accent: "#818cf8", border: "#6366f1", bg: "rgba(99,102,241,0.1)" };

    return (
        <div style={{ minHeight: "100vh", background: "#030712", color: "#f1f5f9", fontFamily: "'DM Sans', sans-serif" }}>
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');
        * { box-sizing: border-box; } body { margin: 0; background: #030712; }
        textarea, input { resize: vertical; }
        textarea:focus, input:focus { outline: none; }
        ::-webkit-scrollbar { width: 6px; height: 6px; } ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
      `}</style>

            <div style={{ position: "fixed", inset: 0, backgroundImage: "linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px)", backgroundSize: "40px 40px", pointerEvents: "none" }} />

            <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "32px 24px", position: "relative" }}>

                <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "36px" }}>
                    <button onClick={onBack}
                        style={{ display: "flex", alignItems: "center", gap: "6px", background: "transparent", border: "1px solid #1e293b", color: "#475569", borderRadius: "8px", padding: "8px 14px", cursor: "pointer", fontSize: "12px", fontFamily: "JetBrains Mono, monospace", transition: "all 0.15s" }}
                        onMouseEnter={e => { e.currentTarget.style.borderColor = "#6366f1"; e.currentTarget.style.color = "#818cf8"; }}
                        onMouseLeave={e => { e.currentTarget.style.borderColor = "#1e293b"; e.currentTarget.style.color = "#475569"; }}>
                        ← Back
                    </button>
                    <div>
                        <h1 style={{ margin: 0, fontSize: "clamp(22px,4vw,36px)", fontFamily: "'DM Serif Display', serif", letterSpacing: "-0.02em", background: "linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                            Corpus Retrieval Demo
                        </h1>
                    </div>
                </div>

                <div style={{ background: "#080f1a", border: "1px solid #1e293b", borderRadius: "20px", padding: "28px", marginBottom: "32px", boxShadow: "0 4px 40px rgba(0,0,0,0.4)" }}>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "16px", alignItems: "end", marginBottom: "20px" }}>
                        <div>
                            <label style={{ display: "block", fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color: "#6366f1", letterSpacing: "0.1em", marginBottom: "8px" }}>QUERY</label>
                            <input value={query} onChange={e => setQuery(e.target.value)}
                                placeholder="Enter your search query…"
                                style={{ width: "100%", background: "#0f172a", border: "1px solid #1e293b", borderRadius: "12px", padding: "13px 16px", color: "#e2e8f0", fontFamily: "'DM Sans', sans-serif", fontSize: "14px", transition: "border-color 0.2s" }}
                                onFocus={e => e.target.style.borderColor = "#6366f1"}
                                onBlur={e => e.target.style.borderColor = "#1e293b"} />
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: "8px", alignItems: "flex-end" }}>
                            <div style={{ fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color: "#475569" }}>DOCUMENTS</div>
                            <div style={{ display: "flex", gap: "4px" }}>
                                {[5, 8, 10, 15, 20].map(n => (
                                    <button key={n} onClick={() => handleNDocsChange(n)}
                                        style={{ padding: "6px 11px", borderRadius: "7px", border: `1px solid ${nDocs === n ? "#6366f1" : "#1e293b"}`, background: nDocs === n ? "rgba(99,102,241,0.15)" : "transparent", color: nDocs === n ? "#818cf8" : "#374151", fontSize: "12px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", transition: "all 0.15s" }}>
                                        {n}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: "10px", marginBottom: "22px" }}>
                        {Array.from({ length: nDocs }, (_, i) => (
                            <div key={i}>
                                <label style={{ display: "block", fontSize: "9px", fontFamily: "JetBrains Mono, monospace", color: "#334155", letterSpacing: "0.1em", marginBottom: "5px" }}>
                                    DOC {i + 1}
                                </label>
                                <textarea value={docs[i] || ""} onChange={e => handleDocChange(i, e.target.value)}
                                    placeholder={`Document ${i + 1}…`} rows={3}
                                    style={{ width: "100%", background: "#0a1628", border: "1px solid #1e293b", borderRadius: "10px", padding: "10px 12px", color: "#cbd5e1", fontFamily: "'DM Sans', sans-serif", fontSize: "13px", lineHeight: "1.5", transition: "border-color 0.2s" }}
                                    onFocus={e => e.target.style.borderColor = "#4f46e5"}
                                    onBlur={e => e.target.style.borderColor = "#1e293b"} />
                            </div>
                        ))}
                    </div>

                    <div style={{ borderTop: "1px solid #1e293b", paddingTop: "16px", marginBottom: "18px" }}>
                        <div style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em", marginBottom: "10px" }}>TEXT PREPROCESSING</div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
                            {[
                                { label: "Lowercase", active: ppLowercase, set: setPpLowercase },
                                { label: "Strip punct.", active: ppPunct, set: setPpPunct },
                                { label: "Remove stopwords", active: ppStopwords, set: setPpStopwords },
                            ].map(({ label, active, set }) => (
                                <button key={label} onClick={() => set(v => !v)}
                                    style={{ display: "flex", alignItems: "center", gap: "7px", padding: "6px 12px", borderRadius: "999px", border: `1px solid ${active ? "rgba(99,102,241,0.6)" : "#1e293b"}`, background: active ? "rgba(99,102,241,0.14)" : "#0a1628", cursor: "pointer", transition: "all 0.18s" }}>
                                    <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: active ? "#818cf8" : "#1e293b", border: `1px solid ${active ? "#6366f1" : "#334155"}`, boxShadow: active ? "0 0 6px #6366f1" : "none", transition: "all 0.18s" }} />
                                    <span style={{ fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color: active ? "#c7d2fe" : "#475569" }}>{label}</span>
                                </button>
                            ))}
                            {ppStopwords && (
                                <div style={{ position: "relative" }}>
                                    <select value={ppLang} onChange={e => setPpLang(e.target.value)}
                                        style={{ appearance: "none", WebkitAppearance: "none", background: "#0a1628", border: "1px solid #6366f1", color: "#c7d2fe", borderRadius: "8px", padding: "5px 24px 5px 10px", fontSize: "11px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", outline: "none" }}>
                                        {spacyLanguages.length > 0
                                            ? spacyLanguages.map(l => (
                                                <option key={l.model} value={l.model} disabled={!l.installed}>
                                                    {l.label}{l.installed ? "" : " (not installed)"}
                                                </option>
                                            ))
                                            : <option value="en_core_web_sm">English</option>
                                        }
                                    </select>
                                    <span style={{ position: "absolute", right: "7px", top: "50%", transform: "translateY(-50%)", color: "#6366f1", fontSize: "9px", pointerEvents: "none" }}>▼</span>
                                </div>
                            )}
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "16px", alignItems: "end" }}>
                        <div style={{ minWidth: 0 }}>
                            {modelGroups.length > 0
                                ? <ModelSelector groups={modelGroups} selectedId={selectedModel} loadedId={loadedModelId} onChange={setSelectedModel} />
                                : <div style={{ fontSize: "12px", color: "#374151", fontFamily: "JetBrains Mono, monospace" }}>⚠ Cannot reach backend</div>
                            }
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: "8px", alignItems: "flex-end" }}>
                            <button onClick={loadExample}
                                style={{ background: "transparent", border: "1px solid #1e293b", color: "#374151", borderRadius: "8px", padding: "7px 14px", fontSize: "11px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", transition: "all 0.15s" }}
                                onMouseEnter={e => { e.currentTarget.style.borderColor = "#334155"; e.currentTarget.style.color = "#64748b"; }}
                                onMouseLeave={e => { e.currentTarget.style.borderColor = "#1e293b"; e.currentTarget.style.color = "#374151"; }}>
                                Load example
                            </button>
                            <button onClick={retrieve} disabled={loading}
                                style={{ background: loading ? "#1e293b" : "linear-gradient(135deg, #6366f1, #8b5cf6)", border: "none", color: loading ? "#475569" : "white", borderRadius: "12px", padding: "13px 36px", fontSize: "14px", fontFamily: "'DM Sans', sans-serif", fontWeight: 600, cursor: loading ? "not-allowed" : "pointer", transition: "all 0.2s", whiteSpace: "nowrap", boxShadow: loading ? "none" : "0 4px 20px rgba(99,102,241,0.4)" }}>
                                {loading
                                    ? <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                        <span style={{ width: "12px", height: "12px", border: "2px solid #475569", borderTopColor: "#6366f1", borderRadius: "50%", display: "inline-block", animation: "spin 0.8s linear infinite" }} />
                                        Retrieving…
                                    </span>
                                    : "Retrieve & Rank →"}
                            </button>
                        </div>
                    </div>

                    {error && <div style={{ marginTop: "12px", color: "#f87171", fontSize: "13px", fontFamily: "JetBrains Mono, monospace" }}>⚠ {error}</div>}
                </div>

                {results && (
                    <div style={{ animation: "fadeIn 0.35s ease" }}>

                        <div style={{ display: "flex", gap: "12px", marginBottom: "24px", flexWrap: "wrap", alignItems: "center" }}>
                            <div style={{ background: "#080f1a", border: "1px solid #1e293b", borderRadius: "10px", padding: "10px 16px", display: "flex", gap: "16px" }}>
                                <div>
                                    <div style={{ fontSize: "9px", color: "#6366f1", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em" }}>DOCUMENTS</div>
                                    <div style={{ fontSize: "20px", fontFamily: "JetBrains Mono, monospace", color: "#f1f5f9", fontWeight: 700 }}>{results.n_docs}</div>
                                </div>
                            </div>

                            <div style={{ display: "flex", gap: "2px", background: "#080f1a", border: "1px solid #1e293b", borderRadius: "10px", padding: "3px", marginLeft: "auto" }}>
                                {[{ id: "ranked", label: "⊞ Ranked view" }, { id: "matrix", label: "⊟ Rank matrix" }].map(v => (
                                    <button key={v.id} onClick={() => setViewMode(v.id)}
                                        style={{ padding: "8px 16px", border: "none", borderRadius: "7px", fontSize: "12px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", transition: "all 0.15s", background: viewMode === v.id ? "rgba(99,102,241,0.2)" : "transparent", color: viewMode === v.id ? "#818cf8" : "#374151" }}>
                                        {v.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {viewMode === "ranked" && (
                            <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: "20px", alignItems: "start" }}>

                                <div style={{ background: "#080f1a", border: "1px solid #1e293b", borderRadius: "16px", padding: "14px", position: "sticky", top: "20px", maxHeight: "80vh", overflowY: "auto" }}>
                                    <div style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em", marginBottom: "12px" }}>SELECT METHOD</div>
                                    {methodGroups.map(([type, methods]) => {
                                        const c = TYPE_COLORS[type] || { accent: "#818cf8", border: "#6366f1" };
                                        return (
                                            <div key={type} style={{ marginBottom: "12px" }}>
                                                <div style={{ fontSize: "9px", color: c.accent, fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.08em", marginBottom: "5px", paddingLeft: "6px", borderLeft: `2px solid ${c.border}` }}>
                                                    {type.toUpperCase().replace(" / ", "/")}
                                                </div>
                                                {methods.map(mkey => (
                                                    <button key={mkey} onClick={() => setSelectedMethod(mkey)}
                                                        style={{ width: "100%", textAlign: "left", padding: "6px 10px", borderRadius: "7px", border: "none", background: selectedMethod === mkey ? c.bg : "transparent", color: selectedMethod === mkey ? c.accent : "#4b5563", fontSize: "11px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", transition: "all 0.15s", display: "block", marginBottom: "2px" }}
                                                        onMouseEnter={e => { if (selectedMethod !== mkey) e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}
                                                        onMouseLeave={e => { if (selectedMethod !== mkey) e.currentTarget.style.background = "transparent"; }}>
                                                        {METHOD_NAMES[mkey]}
                                                    </button>
                                                ))}
                                            </div>
                                        );
                                    })}
                                </div>

                                <div>
                                    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
                                        <span style={{ fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color: selectedColors.accent, background: selectedColors.bg, border: `1px solid ${selectedColors.border}`, borderRadius: "6px", padding: "3px 10px" }}>
                                            {METHOD_TYPES[selectedMethod]}
                                        </span>
                                        <h2 style={{ margin: 0, fontSize: "18px", fontFamily: "'DM Serif Display', serif", color: "#f1f5f9" }}>
                                            {METHOD_NAMES[selectedMethod]}
                                        </h2>
                                    </div>

                                    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                        {(results.rankings[selectedMethod] || []).map((entry, i) => {
                                            const isHighlighted = highlightDoc === entry.doc_index;
                                            const docText = docs[entry.doc_index]?.trim() || "(empty)";
                                            return (
                                                <div key={entry.doc_index}
                                                    onMouseEnter={() => setHighlightDoc(entry.doc_index)}
                                                    onMouseLeave={() => setHighlightDoc(null)}
                                                    style={{ background: isHighlighted ? "rgba(99,102,241,0.06)" : "#080f1a", border: `1px solid ${isHighlighted ? selectedColors.border : "#1e293b"}`, borderRadius: "12px", padding: "14px 18px", transition: "all 0.2s", animation: `fadeIn ${0.1 + i * 0.04}s ease both` }}>
                                                    <div style={{ display: "flex", alignItems: "flex-start", gap: "12px" }}>
                                                        <RankBadge rank={entry.rank} total={results.n_docs} />
                                                        <div style={{ flex: 1, minWidth: 0 }}>
                                                            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                                                                <span style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace" }}>DOC {entry.doc_index + 1}</span>
                                                            </div>
                                                            <p style={{ margin: "0 0 10px", fontSize: "13px", color: "#94a3b8", lineHeight: "1.6", fontFamily: "'DM Sans', sans-serif" }}>
                                                                {docText.slice(0, 180)}{docText.length > 180 ? "…" : ""}
                                                            </p>
                                                            <ScoreBar score={entry.score} color={selectedColors.accent} />
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        )}

                        {viewMode === "matrix" && (
                            <div>
                                <div style={{ marginBottom: "16px" }}>
                                    <div style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em", marginBottom: "10px" }}>
                                        FILTER METHODS  <span style={{ color: "#1e293b" }}>({activeMethods.length} of {ALL_METHODS.length} shown)</span>
                                    </div>
                                    <div style={{ display: "flex", flexWrap: "wrap", gap: "5px" }}>
                                        {ALL_METHODS.map(mkey => {
                                            const c = TYPE_COLORS[METHOD_TYPES[mkey]] || { accent: "#818cf8", border: "#6366f1", bg: "rgba(99,102,241,0.1)" };
                                            const active = activeMethods.includes(mkey);
                                            return (
                                                <button key={mkey} onClick={() => toggleMethod(mkey)}
                                                    style={{ padding: "4px 10px", borderRadius: "6px", border: `1px solid ${active ? c.border : "#1e293b"}`, background: active ? c.bg : "transparent", color: active ? c.accent : "#334155", fontSize: "10px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", transition: "all 0.15s" }}>
                                                    {METHOD_NAMES[mkey]}
                                                </button>
                                            );
                                        })}
                                        <button onClick={() => setActiveMethods(ALL_METHODS)}
                                            style={{ padding: "4px 10px", borderRadius: "6px", border: "1px solid #1e293b", background: "transparent", color: "#374151", fontSize: "10px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer" }}>
                                            All
                                        </button>
                                        <button onClick={() => setActiveMethods(["embedding_cosine", "wmd", "bert_score", "soft_cosine"])}
                                            style={{ padding: "4px 10px", borderRadius: "6px", border: "1px solid #1e293b", background: "transparent", color: "#374151", fontSize: "10px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer" }}>
                                            Semantic only
                                        </button>
                                        <button onClick={() => setActiveMethods(["jaccard", "dice", "overlap", "containment", "tfidf_cosine", "bigram_overlap", "lcs"])}
                                            style={{ padding: "4px 10px", borderRadius: "6px", border: "1px solid #1e293b", background: "transparent", color: "#374151", fontSize: "10px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer" }}>
                                            Lexical only
                                        </button>
                                    </div>
                                </div>

                                <DivergenceTable
                                    rankings={results.rankings}
                                    docLabels={docLabels}
                                    selectedMethods={activeMethods.filter(k => results.rankings[k])}
                                />
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}