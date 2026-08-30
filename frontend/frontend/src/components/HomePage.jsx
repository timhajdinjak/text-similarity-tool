import { useState, useCallback, useEffect } from "react";
import ModelSelector from "./ModelSelector";
import SimilarityCard from "./SimilarityCard";
import EmbeddingModal from "./EmbeddingModal";
import { InfoModal } from "./UIComponents";
import RobustnessModal from "../RobustnessModal";
import { API_URL, TYPE_COLORS, DEFAULT_METHODS } from "../data/constants";

export default function HomePage({ onGoCorpus }) {
    const [text1, setText1] = useState("");
    const [text2, setText2] = useState("");
    const [methods, setMethods] = useState(DEFAULT_METHODS);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [infoModal, setInfoModal] = useState(null);
    const [showEmbeddings, setShowEmbeddings] = useState(false);
    const [showRobustness, setShowRobustness] = useState(false);
    const [submitted, setSubmitted] = useState({ t1: "", t2: "", t1Pre: "", t2Pre: "", modelId: "all-MiniLM-L6-v2", ppOpts: {} });

    const [modelGroups, setModelGroups] = useState([]);
    const [selectedModelId, setSelectedModelId] = useState("all-MiniLM-L6-v2");
    const [loadedModelId, setLoadedModelId] = useState(null);
    const [lastUsedModel, setLastUsedModel] = useState(null);

    const [ppLowercase, setPpLowercase] = useState(true);
    const [ppPunct, setPpPunct] = useState(true);
    const [ppStopwords, setPpStopwords] = useState(false);
    const [ppLang, setPpLang] = useState("en_core_web_sm");
    const [spacyLanguages, setSpacyLanguages] = useState([]);
    const [loadingPhase, setLoadingPhase] = useState("");

    const [previewT1, setPreviewT1] = useState("");
    const [previewT2, setPreviewT2] = useState("");
    const [showPreview, setShowPreview] = useState(false);

    const ppOpts = {
        lowercase: ppLowercase,
        remove_punct: ppPunct,
        remove_stopwords: ppStopwords,
        spacy_model: ppLang,
    };

    useEffect(() => {
        fetch(`${API_URL}/models`)
            .then(r => r.json())
            .then(d => {
                setModelGroups(d.groups || []);
                if (d.default_model) setSelectedModelId(d.default_model);
                if (d.loaded_model) setLoadedModelId(d.loaded_model);
            })
            .catch(() => { });
    }, []);

    useEffect(() => {
        fetch(`${API_URL}/spacy-models`)
            .then(r => r.json())
            .then(d => setSpacyLanguages(d.languages || []))
            .catch(() => { });
    }, []);

    const analyze = useCallback(async (t1, t2, modelId, opts) => {
        if (!t1.trim() || !t2.trim()) { setMethods(DEFAULT_METHODS); return; }
        setLoading(true);
        setError(null);

        const anyPreprocessing = opts.remove_stopwords || opts.remove_punct || !opts.lowercase;
        setLoadingPhase(anyPreprocessing ? "preprocessing" : "analyzing");

        try {
            const res = await fetch(`${API_URL}/similarity`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text1: t1, text2: t2, model_id: modelId, ...opts }),
            });
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.error || `Server error: ${res.status}`);
            }
            setLoadingPhase("analyzing");
            const data = await res.json();
            setMethods(prev => prev.map(m => {
                const r = data.results[m.key];
                if (!r) return m;
                const update = { ...m, score: r.score, highlights_text1: r.highlights_text1 || [], highlights_text2: r.highlights_text2 || [] };
                if (m.key === "embedding_cosine") {
                    update.anisotropy_before = r.anisotropy_before ?? null;
                }
                return update;
            }));
            setSubmitted({ t1, t2, t1Pre: data.text1, t2Pre: data.text2, modelId, ppOpts: opts });
            setLoadedModelId(data.loaded_model || modelId);
            setLastUsedModel(data.loaded_model || data.model_id);
            setPreviewT1(data.text1 || t1);
            setPreviewT2(data.text2 || t2);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
            setLoadingPhase("");
        }
    }, []);

    const handleSubmit = () => analyze(text1, text2, selectedModelId, ppOpts);
    const handleKeyDown = (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleSubmit(); };

    const [origText2, setOrigText2] = useState(null);
    const [mutationApplied, setMutationApplied] = useState("");

    const rng = () => Math.random();
    const pick = (arr) => arr[Math.floor(rng() * arr.length)];
    const randInt = (lo, hi) => Math.floor(rng() * (hi - lo + 1)) + lo;

    const applyMutation = useCallback((type) => {
        if (type === "reset") {
            if (origText2 !== null) { setText2(origText2); setOrigText2(null); setMutationApplied(""); }
            return;
        }
        if (!text2.trim()) return;

        if (origText2 === null) setOrigText2(text2);

        let result = "";
        let label = "";

        if (type === "shuffle") {
            const tokens = text2.split(/(\s+)/);
            const words = tokens.filter((_, i) => i % 2 === 0);
            for (let i = words.length - 1; i > 0; i--) {
                const j = Math.floor(rng() * (i + 1));
                [words[i], words[j]] = [words[j], words[i]];
            }
            result = words.join(" ");
            label = "word order shuffled";
        }

        else if (type === "typos") {
            const TYPO_RATE = 0.25;
            const SUBS = {
                a: ["@", "4", "q", "s"], e: ["3", "€", "a", "r"], i: ["1", "!", "l", "j"],
                o: ["0", "oo", "p", "u"], l: ["1", "ll", "i"], s: ["$", "5", "z"],
                t: ["tt", "7", "y"], n: ["m", "nn", "h"], r: ["rr", "4", "t"],
                c: ["k", "cc", "v"], u: ["uu", "v", "y"], g: ["9", "gg", "q"],
                h: ["hh", "n", "j"], m: ["mm", "n", "w"], p: ["pp", "b", "o"],
                f: ["ff", "ph", "v"], b: ["bb", "p", "v"], d: ["dd", "t", "c"],
                k: ["kk", "c", "q"], w: ["ww", "v", "q"], y: ["yy", "i", "u"],
            };
            const DOUBLE_CONSONANTS = ["b", "d", "f", "g", "l", "m", "n", "p", "r", "s", "t", "z"];
            result = text2.split(" ").map(word => {
                if (word.length < 2 || rng() > TYPO_RATE) return word;
                const typoType = pick(["sub", "del", "double", "transpose", "insert"]);
                const chars = word.split("");
                const idx = randInt(0, chars.length - 1);
                const ch = chars[idx].toLowerCase();
                if (typoType === "sub" && SUBS[ch]) {
                    chars[idx] = pick(SUBS[ch]);
                } else if (typoType === "del" && word.length > 2) {
                    chars.splice(idx, 1);
                } else if (typoType === "double") {
                    const dc = pick(DOUBLE_CONSONANTS);
                    const pos = randInt(0, chars.length - 1);
                    chars.splice(pos, 0, chars[pos]);
                } else if (typoType === "transpose" && idx < chars.length - 1) {
                    [chars[idx], chars[idx + 1]] = [chars[idx + 1], chars[idx]];
                } else if (typoType === "insert") {
                    const extras = "aeiourstln";
                    chars.splice(idx, 0, pick(extras.split("")));
                }
                return chars.join("");
            }).join(" ");
            const changed = result.split(" ").filter((w, i) => w !== text2.split(" ")[i]).length;
            label = `${changed} typo${changed !== 1 ? "s" : ""} introduced`;
        }

        else if (type === "noise") {
            const NOISE_RATE = 0.30;
            const NOISE_WORDS = ["um", "uh", "like", "er", "hmm", "xyz", "foo", "bar", "blah", "qux", "zzz", "etc", "...", "???", "###"];
            const NOISE_CHARS = ["@", "#", "$", "%", "&", "*", "!", "~", "^", "_", "|", "<", ">", "[", "]", "{", "}"];
            const NOISE_NUMS = () => String(randInt(1, 9999));

            const tokens = text2.split(" ");
            const noisy = [];
            let insertCount = 0;

            const swapped = [...tokens];
            const swaps = randInt(1, Math.max(1, Math.floor(tokens.length / 6)));
            for (let s = 0; s < swaps; s++) {
                const i = randInt(0, swapped.length - 1);
                const j = randInt(0, swapped.length - 1);
                [swapped[i], swapped[j]] = [swapped[j], swapped[i]];
            }

            swapped.forEach(word => {
                noisy.push(word);
                if (rng() < NOISE_RATE) {
                    const noiseType = pick(["word", "char", "num", "punct"]);
                    if (noiseType === "word") { noisy.push(pick(NOISE_WORDS)); insertCount++; }
                    if (noiseType === "char") { noisy.push(pick(NOISE_CHARS)); insertCount++; }
                    if (noiseType === "num") { noisy.push(NOISE_NUMS()); insertCount++; }
                    if (noiseType === "punct") {
                        const puncts = ["...", "!!", "??", ";;", "--", ",,"];
                        noisy.push(pick(puncts)); insertCount++;
                    }
                }
            });
            result = noisy.join(" ");
            label = `${insertCount} noise token${insertCount !== 1 ? "s" : ""} + ${swaps} swap${swaps !== 1 ? "s" : ""}`;
        }

        setText2(result);
        setMutationApplied(label);
    }, [text2, origText2]);

    const groupedMethods = DEFAULT_METHODS.reduce((acc, m) => {
        if (!acc[m.type]) acc[m.type] = [];
        acc[m.type].push(methods.find(x => x.key === m.key) || m);
        return acc;
    }, {});

    const hasResults = methods.some(m => m.score !== null);
    const avgScore = hasResults ? methods.filter(m => m.score !== null).reduce((s, m, _, a) => s + m.score / a.length, 0) : 0;

    const willSwapModel = loadedModelId !== null && loadedModelId !== selectedModelId;

    return (
        <div style={{ minHeight: "100vh", background: "#030712", color: "#f1f5f9", fontFamily: "'DM Sans', sans-serif" }}>
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');
        * { box-sizing: border-box; } body { margin: 0; background: #030712; }
        textarea { resize: vertical; } textarea:focus { outline: none; }
        ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
      `}</style>
            <div style={{ position: "fixed", inset: 0, backgroundImage: "linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px)", backgroundSize: "40px 40px", pointerEvents: "none" }} />

            <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "48px 24px", position: "relative" }}>

                <div style={{ textAlign: "center", marginBottom: "52px" }}>
                    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "10px", marginBottom: "20px", flexWrap: "wrap" }}>
                        <button onClick={onGoCorpus}
                            style={{ display: "inline-flex", alignItems: "center", gap: "7px", background: "rgba(139,92,246,0.1)", border: "1px solid rgba(139,92,246,0.35)", borderRadius: "20px", padding: "6px 16px", cursor: "pointer", transition: "all 0.18s" }}
                            onMouseEnter={e => { e.currentTarget.style.background = "rgba(139,92,246,0.22)"; e.currentTarget.style.borderColor = "#8b5cf6"; }}
                            onMouseLeave={e => { e.currentTarget.style.background = "rgba(139,92,246,0.1)"; e.currentTarget.style.borderColor = "rgba(139,92,246,0.35)"; }}>
                            <span style={{ fontSize: "12px", fontFamily: "JetBrains Mono, monospace", color: "#a78bfa", letterSpacing: "0.1em" }}>CORPUS DEMO</span>
                        </button>
                    </div>
                    <h1 style={{ margin: "0 0 12px", fontSize: "clamp(32px,5vw,52px)", fontFamily: "'DM Serif Display', serif", letterSpacing: "-0.02em", background: "linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                        Text Similarity Tool
                    </h1>
                </div>

                <div style={{ background: "#080f1a", border: "1px solid #1e293b", borderRadius: "20px", padding: "28px", marginBottom: "40px", boxShadow: "0 4px 40px rgba(0,0,0,0.4)" }}>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "20px" }}>

                        <div>
                            <label style={{ display: "block", fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color: "#6366f1", letterSpacing: "0.1em", marginBottom: "8px" }}>TEXT A</label>
                            <textarea value={text1} onChange={e => setText1(e.target.value)} onKeyDown={handleKeyDown}
                                placeholder="Enter text a here…" rows={5}
                                style={{ width: "100%", background: "#0f172a", border: "1px solid #1e293b", borderRadius: "12px", padding: "14px", color: "#e2e8f0", fontFamily: "'DM Sans', sans-serif", fontSize: "14px", lineHeight: "1.6", transition: "border-color 0.2s" }}
                                onFocus={e => e.target.style.borderColor = "#6366f1"}
                                onBlur={e => e.target.style.borderColor = "#1e293b"} />
                        </div>

                        <div>
                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
                                <label style={{ fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color: "#6366f1", letterSpacing: "0.1em" }}>TEXT B</label>
                                <div style={{ display: "flex", gap: "5px" }}>
                                    {[
                                        { id: "shuffle", icon: "⇄", label: "Shuffle words", title: "Randomly shuffle word order" },
                                        { id: "typos", icon: "✗", label: "Add typos", title: "Introduce random typos (substitutions, deletions, doublings)" },
                                        { id: "noise", icon: "~", label: "Add noise", title: "Insert random words, symbols and numbers" },
                                        { id: "reset", icon: "↺", label: "Reset", title: "Restore original text" },
                                    ].map(({ id, icon, label, title }) => (
                                        <button key={id} title={title}
                                            disabled={!text2.trim() && id !== "reset"}
                                            onClick={() => applyMutation(id)}
                                            style={{
                                                display: "flex", alignItems: "center", gap: "4px",
                                                padding: "4px 9px", borderRadius: "6px", fontSize: "11px",
                                                fontFamily: "JetBrains Mono, monospace",
                                                border: `1px solid ${id === "reset" ? "#1e293b" : "rgba(139,92,246,0.35)"}`,
                                                background: id === "reset" ? "transparent" : "rgba(139,92,246,0.1)",
                                                color: id === "reset" ? "#334155" : "#a78bfa",
                                                cursor: text2.trim() || id === "reset" ? "pointer" : "not-allowed",
                                                opacity: !text2.trim() && id !== "reset" ? 0.4 : 1,
                                                transition: "all 0.15s",
                                            }}
                                            onMouseEnter={e => { if (text2.trim() || id === "reset") { e.currentTarget.style.background = id === "reset" ? "rgba(255,255,255,0.04)" : "rgba(139,92,246,0.22)"; e.currentTarget.style.borderColor = id === "reset" ? "#334155" : "#7c3aed"; } }}
                                            onMouseLeave={e => { e.currentTarget.style.background = id === "reset" ? "transparent" : "rgba(139,92,246,0.1)"; e.currentTarget.style.borderColor = id === "reset" ? "#1e293b" : "rgba(139,92,246,0.35)"; }}
                                        >
                                            <span style={{ fontSize: "12px" }}>{icon}</span>
                                            <span>{label}</span>
                                        </button>
                                    ))}
                                </div>
                            </div>
                            <textarea value={text2} onChange={e => { setText2(e.target.value); setOrigText2(null); }} onKeyDown={handleKeyDown}
                                placeholder="Enter text b here…" rows={5}
                                style={{ width: "100%", background: "#0f172a", border: "1px solid #1e293b", borderRadius: "12px", padding: "14px", color: "#e2e8f0", fontFamily: "'DM Sans', sans-serif", fontSize: "14px", lineHeight: "1.6", transition: "border-color 0.2s" }}
                                onFocus={e => e.target.style.borderColor = "#6366f1"}
                                onBlur={e => e.target.style.borderColor = "#1e293b"} />
                        </div>
                    </div>

                    <div style={{ borderTop: "1px solid #1e293b", marginTop: "20px", paddingTop: "18px", display: "flex", flexDirection: "column", gap: "12px" }}>
                        <div style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em" }}>TEXT PREPROCESSING</div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "flex-start" }}>

                            {[
                                { label: "Lowercase", active: ppLowercase, set: setPpLowercase, desc: "Fold all characters to lowercase before comparison" },
                                { label: "Strip punctuation", active: ppPunct, set: setPpPunct, desc: "Remove all punctuation marks and symbols" },
                                { label: "Remove stopwords", active: ppStopwords, set: setPpStopwords, desc: "Remove common function words via spaCy" },
                            ].map(({ label, active, set, desc }) => (
                                <button key={label} onClick={() => set(v => !v)} title={desc}
                                    style={{ display: "flex", alignItems: "center", gap: "8px", padding: "7px 14px", borderRadius: "999px", border: `1px solid ${active ? "rgba(99,102,241,0.6)" : "#1e293b"}`, background: active ? "rgba(99,102,241,0.14)" : "#0a1628", cursor: "pointer", transition: "all 0.18s", userSelect: "none" }}>
                                    <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: active ? "#818cf8" : "#1e293b", border: `1px solid ${active ? "#6366f1" : "#334155"}`, boxShadow: active ? "0 0 6px #6366f1" : "none", transition: "all 0.18s", flexShrink: 0 }} />
                                    <span style={{ fontSize: "12px", fontFamily: "JetBrains Mono, monospace", color: active ? "#c7d2fe" : "#475569", letterSpacing: "0.03em" }}>{label}</span>
                                </button>
                            ))}

                            {ppStopwords && (
                                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                    <span style={{ fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color: "#475569" }}>language:</span>
                                    <div style={{ position: "relative" }}>
                                        <select value={ppLang} onChange={e => setPpLang(e.target.value)}
                                            style={{ appearance: "none", WebkitAppearance: "none", background: "#0a1628", border: "1px solid #6366f1", color: "#c7d2fe", borderRadius: "8px", padding: "6px 28px 6px 12px", fontSize: "12px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", outline: "none" }}>
                                            {spacyLanguages.length > 0
                                                ? spacyLanguages.map(l => (
                                                    <option key={l.model} value={l.model} disabled={!l.installed}
                                                        style={{ background: "#0a1628", color: l.installed ? "#c7d2fe" : "#374151" }}>
                                                        {l.label}{l.installed ? "" : " (not installed)"}
                                                    </option>
                                                ))
                                                : <option value="en_core_web_sm">English</option>
                                            }
                                        </select>
                                        <span style={{ position: "absolute", right: "8px", top: "50%", transform: "translateY(-50%)", color: "#6366f1", fontSize: "10px", pointerEvents: "none" }}>▼</span>
                                    </div>
                                    {spacyLanguages.find(l => l.model === ppLang && !l.installed) && (
                                        <span style={{ fontSize: "10px", color: "#f59e0b", fontFamily: "JetBrains Mono, monospace" }}>
                                            ⚠ run: python -m spacy download {ppLang}
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>

                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "16px", alignItems: "end", marginTop: "18px" }}>
                        <div style={{ minWidth: 0 }}>
                            {modelGroups.length > 0
                                ? <ModelSelector groups={modelGroups} selectedId={selectedModelId} loadedId={loadedModelId} onChange={setSelectedModelId} />
                                : <div style={{ fontSize: "12px", color: "#374151", fontFamily: "JetBrains Mono, monospace", paddingTop: "8px" }}>⚠ Cannot reach backend - model list unavailable</div>
                            }
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "8px" }}>
                            {willSwapModel && (
                                <div style={{ fontSize: "11px", color: "#f59e0b", fontFamily: "JetBrains Mono, monospace", display: "flex", alignItems: "center", gap: "6px" }}>
                                    <span>⟳</span> model will reload on submit
                                </div>
                            )}
                            <button onClick={handleSubmit} disabled={loading || !text1.trim() || !text2.trim()}
                                style={{ background: loading ? "#1e293b" : "linear-gradient(135deg, #6366f1, #8b5cf6)", border: "none", color: loading ? "#475569" : "white", borderRadius: "12px", padding: "12px 32px", fontSize: "14px", fontFamily: "'DM Sans', sans-serif", fontWeight: 600, cursor: loading ? "not-allowed" : "pointer", transition: "all 0.2s", whiteSpace: "nowrap", boxShadow: loading ? "none" : "0 4px 20px rgba(99,102,241,0.4)" }}>
                                {loading ? (
                                    <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                        <span style={{ width: "12px", height: "12px", border: "2px solid #475569", borderTopColor: "#6366f1", borderRadius: "50%", display: "inline-block", animation: "spin 0.8s linear infinite" }} />
                                        {loadingPhase === "preprocessing" ? "Preprocessing…" : willSwapModel ? "Loading model…" : "Analyzing…"}
                                    </span>
                                ) : "Analyze →"}
                            </button>
                        </div>
                    </div>

                    {lastUsedModel && (
                        <div style={{ marginTop: "16px", paddingTop: "16px", borderTop: "1px solid #1e293b" }}>

                            <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                                <span style={{ fontSize: "11px", color: "#374151", fontFamily: "JetBrains Mono, monospace" }}>Last analysis:</span>
                                <span style={{ background: "rgba(16,185,129,0.1)", border: "1px solid #10b981", color: "#34d399", borderRadius: "6px", padding: "2px 10px", fontSize: "11px", fontFamily: "JetBrains Mono, monospace" }}>
                                    {lastUsedModel}
                                </span>
                                {loadedModelId && loadedModelId !== lastUsedModel && (
                                    <span style={{ fontSize: "11px", color: "#475569", fontFamily: "JetBrains Mono, monospace" }}>
                                        · in RAM: <span style={{ color: "#fbbf24" }}>{loadedModelId}</span>
                                    </span>
                                )}

                                {(previewT1 !== submitted.t1 || previewT2 !== submitted.t2) && (
                                    <button onClick={() => setShowPreview(v => !v)}
                                        style={{ display: "flex", alignItems: "center", gap: "6px", background: showPreview ? "rgba(245,158,11,0.15)" : "rgba(245,158,11,0.07)", border: "1px solid rgba(245,158,11,0.35)", color: "#fbbf24", borderRadius: "7px", padding: "3px 10px", fontSize: "11px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer" }}>
                                        <span>◈</span> {showPreview ? "Hide" : "Show"} preprocessed text
                                    </button>
                                )}

                                <button
                                    onClick={() => setShowEmbeddings(true)}
                                    style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "7px", background: "linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.12))", border: "1px solid rgba(99,102,241,0.35)", color: "#818cf8", borderRadius: "9px", padding: "6px 16px", fontSize: "12px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", letterSpacing: "0.05em", transition: "all 0.2s" }}
                                    onMouseEnter={e => { e.currentTarget.style.background = "linear-gradient(135deg, rgba(99,102,241,0.28), rgba(139,92,246,0.28))"; e.currentTarget.style.borderColor = "#6366f1"; }}
                                    onMouseLeave={e => { e.currentTarget.style.background = "linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.12))"; e.currentTarget.style.borderColor = "rgba(99,102,241,0.35)"; }}
                                >
                                    <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
                                        <circle cx="3" cy="3" r="2.5" fill="#818cf8" />
                                        <circle cx="13" cy="5" r="2.5" fill="#34d399" />
                                        <circle cx="6" cy="12" r="2.5" fill="#818cf8" />
                                        <circle cx="11" cy="11" r="2.5" fill="#34d399" />
                                        <circle cx="8" cy="7" r="1.8" fill="#c084fc" />
                                    </svg>
                                    Show Embedding Space
                                </button>
                                <button
                                    onClick={() => setShowRobustness(true)}
                                    style={{ display: "flex", alignItems: "center", gap: "7px", background: "linear-gradient(135deg, rgba(245,158,11,0.10), rgba(239,68,68,0.10))", border: "1px solid rgba(245,158,11,0.35)", color: "#fbbf24", borderRadius: "9px", padding: "6px 16px", fontSize: "12px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", letterSpacing: "0.05em", transition: "all 0.2s" }}
                                    onMouseEnter={e => { e.currentTarget.style.background = "linear-gradient(135deg, rgba(245,158,11,0.24), rgba(239,68,68,0.24))"; e.currentTarget.style.borderColor = "#f59e0b"; }}
                                    onMouseLeave={e => { e.currentTarget.style.background = "linear-gradient(135deg, rgba(245,158,11,0.10), rgba(239,68,68,0.10))"; e.currentTarget.style.borderColor = "rgba(245,158,11,0.35)"; }}
                                >
                                    <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
                                        <polyline points="1,12 4,6 7,9 10,3 15,7" stroke="#fbbf24" strokeWidth="1.8" fill="none" strokeLinejoin="round" strokeLinecap="round" />
                                        <line x1="1" y1="14" x2="15" y2="14" stroke="#f59e0b" strokeWidth="1" opacity="0.4" />
                                    </svg>
                                    Robustness Curve
                                </button>
                            </div>
                        </div>
                    )}

                    {showPreview && lastUsedModel && (
                        <div style={{ marginTop: "14px", background: "#030712", border: "1px solid rgba(245,158,11,0.25)", borderRadius: "12px", padding: "16px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                            <div style={{ fontSize: "9px", color: "#92400e", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em", gridColumn: "1/-1", marginBottom: "4px" }}>PREPROCESSED INPUT - what the similarity methods actually received</div>
                            {[{ label: "TEXT A", val: previewT1 }, { label: "TEXT B", val: previewT2 }].map(({ label, val }) => (
                                <div key={label}>
                                    <div style={{ fontSize: "9px", color: "#78350f", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em", marginBottom: "6px" }}>{label}</div>
                                    <div style={{ background: "#0a0f1a", border: "1px solid #1e293b", borderRadius: "8px", padding: "10px 12px", fontSize: "13px", color: "#fbbf24", fontFamily: "'DM Sans', sans-serif", lineHeight: "1.6", minHeight: "48px", wordBreak: "break-word" }}>
                                        {val || <span style={{ color: "#374151", fontStyle: "italic" }}>empty after preprocessing</span>}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {error && <div style={{ marginTop: "12px", color: "#f87171", fontSize: "13px", fontFamily: "JetBrains Mono, monospace" }}>⚠ {error}</div>}
                </div>

                {hasResults && (
                    <div style={{ display: "flex", gap: "16px", marginBottom: "32px", overflowX: "auto", paddingBottom: "4px" }}>
                        <div style={{ background: "#080f1a", border: "1px solid #1e293b", borderRadius: "12px", padding: "14px 20px", flexShrink: 0 }}>
                            <div style={{ fontSize: "10px", color: "#6366f1", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em", marginBottom: "4px" }}>AVG SIMILARITY</div>
                            <div style={{ fontSize: "24px", fontFamily: "'DM Serif Display', serif", color: "#f1f5f9" }}>{Math.round(avgScore * 100)}%</div>
                        </div>
                        {Object.entries(TYPE_COLORS).map(([type, colors]) => {
                            const typeMeths = methods.filter(m => m.type === type && m.score !== null);
                            if (!typeMeths.length) return null;
                            const avg = typeMeths.reduce((s, m) => s + m.score, 0) / typeMeths.length;
                            return (
                                <div key={type} style={{ background: "#080f1a", border: `1px solid ${colors.border}30`, borderRadius: "12px", padding: "14px 20px", flexShrink: 0 }}>
                                    <div style={{ fontSize: "10px", color: colors.accent, fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em", marginBottom: "4px", maxWidth: "120px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{type.toUpperCase()}</div>
                                    <div style={{ fontSize: "22px", fontFamily: "'DM Serif Display', serif", color: "#f1f5f9" }}>{Math.round(avg * 100)}%</div>
                                </div>
                            );
                        })}
                    </div>
                )}

                {Object.entries(groupedMethods).map(([type, meths]) => {
                    const colors = TYPE_COLORS[type] || { accent: "#818cf8", border: "#6366f1" };
                    return (
                        <div key={type} style={{ marginBottom: "32px" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "14px" }}>
                                <span style={{ width: "3px", height: "20px", background: colors.border, borderRadius: "2px", display: "inline-block" }} />
                                <span style={{ fontSize: "12px", fontFamily: "JetBrains Mono, monospace", color: colors.accent, letterSpacing: "0.1em", textTransform: "uppercase" }}>{type}</span>
                            </div>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: "12px" }}>
                                {meths.map(m => <SimilarityCard key={m.key} method={m} text1={submitted.t1Pre || submitted.t1} text2={submitted.t2Pre || submitted.t2} modelId={submitted.modelId} ppOpts={submitted.ppOpts} onInfo={setInfoModal} />)}
                            </div>
                        </div>
                    );
                })}

            </div>

            {infoModal && <InfoModal method={infoModal} onClose={() => setInfoModal(null)} />}
            {showEmbeddings && (
                <EmbeddingModal
                    text1={submitted.t1}
                    text2={submitted.t2}
                    modelId={submitted.modelId}
                    ppOpts={submitted.ppOpts}
                    onClose={() => setShowEmbeddings(false)}
                />
            )}
            {showRobustness && (
                <RobustnessModal
                    text1={submitted.t1}
                    text2={submitted.t2}
                    modelId={submitted.modelId}
                    ppOpts={submitted.ppOpts}
                    onClose={() => setShowRobustness(false)}
                />
            )}
        </div>
    );
}