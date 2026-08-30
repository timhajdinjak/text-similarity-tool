import { useState } from "react";
import { TYPE_COLORS } from "../data/constants";

export function ScoreRing({ score, color }) {
    const pct = (score ?? 0) * 100;
    const r = 20, circ = 2 * Math.PI * r;
    const dash = (pct / 100) * circ;
    return (
        <svg width="56" height="56" viewBox="0 0 56 56">
            <circle cx="28" cy="28" r={r} fill="none" stroke="#1e293b" strokeWidth="5" />
            <circle cx="28" cy="28" r={r} fill="none" stroke={color} strokeWidth="5"
                strokeDasharray={`${dash} ${circ - dash}`} strokeDashoffset={circ * 0.25} strokeLinecap="round"
                style={{ transition: "stroke-dasharray 0.6s cubic-bezier(.4,0,.2,1)" }} />
            <text x="28" y="33" textAnchor="middle" fill={score === null ? "#374151" : color}
                style={{ fontSize: "11px", fontFamily: "JetBrains Mono, monospace", fontWeight: 700 }}>
                {score === null ? "-" : `${Math.round(pct)}%`}
            </text>
        </svg>
    );
}

export function InfoModal({ method, onClose }) {
    const colors = TYPE_COLORS[method.type] || { accent: "#818cf8", border: "#6366f1", bg: "rgba(99,102,241,0.1)" };
    const [tab, setTab] = useState("explain");

    const tabs = [
        { id: "explain", label: "What it is" },
        { id: "formula", label: "Formula" },
        { id: "example", label: "Example" },
        { id: "guide", label: "When to use" },
    ];

    const SensitivityPill = ({ label, value }) => {
        const positive = value.startsWith("✓");
        const neutral = value.startsWith("~");
        const c = positive ? "#34d399" : neutral ? "#fbbf24" : "#f87171";
        const bg = positive ? "rgba(52,211,153,0.1)" : neutral ? "rgba(251,191,36,0.1)" : "rgba(248,113,113,0.1)";
        return (
            <div style={{ display: "flex", flexDirection: "column", gap: "3px", flex: "1 1 120px" }}>
                <span style={{ fontSize: "9px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.08em", textTransform: "uppercase" }}>{label}</span>
                <span style={{ fontSize: "11px", color: c, background: bg, border: `1px solid ${c}40`, borderRadius: "5px", padding: "3px 7px", fontFamily: "JetBrains Mono, monospace" }}>{value}</span>
            </div>
        );
    };

    return (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.82)", backdropFilter: "blur(8px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, padding: "16px" }}
            onClick={onClose}>
            <div style={{ background: "#07111f", border: `1px solid ${colors.border}`, borderRadius: "20px", width: "min(96vw, 600px)", maxHeight: "88vh", display: "flex", flexDirection: "column", boxShadow: `0 0 60px ${colors.border}30`, overflow: "hidden" }}
                onClick={e => e.stopPropagation()}>

                <div style={{ padding: "22px 24px 16px", borderBottom: "1px solid #0f2236" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                        <div>
                            <span style={{ fontSize: "10px", fontFamily: "JetBrains Mono, monospace", color: colors.accent, letterSpacing: "0.12em", textTransform: "uppercase", background: colors.bg, padding: "3px 8px", borderRadius: "4px", border: `1px solid ${colors.border}` }}>{method.type}</span>
                            <h2 style={{ margin: "10px 0 0", fontSize: "22px", fontFamily: "'DM Serif Display', serif", color: "#f1f5f9" }}>{method.name}</h2>
                        </div>
                        <button onClick={onClose} style={{ background: "transparent", border: "1px solid #1e293b", color: "#475569", borderRadius: "8px", padding: "6px 10px", cursor: "pointer", fontSize: "16px", flexShrink: 0, marginLeft: "16px" }}>✕</button>
                    </div>
                    <div style={{ display: "flex", gap: "2px", marginTop: "16px", background: "#030c18", borderRadius: "8px", padding: "3px" }}>
                        {tabs.map(t => (
                            <button key={t.id} onClick={() => setTab(t.id)}
                                style={{ flex: 1, padding: "7px 4px", border: "none", borderRadius: "6px", fontSize: "12px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", transition: "all 0.15s", letterSpacing: "0.02em", background: tab === t.id ? colors.bg : "transparent", color: tab === t.id ? colors.accent : "#374151", fontWeight: tab === t.id ? 600 : 400, borderBottom: tab === t.id ? `2px solid ${colors.border}` : "2px solid transparent" }}>
                                {t.label}
                            </button>
                        ))}
                    </div>
                </div>

                <div style={{ overflowY: "auto", flex: 1 }}>
                    {tab === "explain" && (
                        <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: "18px" }}>
                            <div style={{ background: "rgba(99,102,241,0.06)", border: "1px solid rgba(99,102,241,0.15)", borderRadius: "12px", padding: "16px" }}>
                                <div style={{ fontSize: "10px", color: "#6366f1", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em", marginBottom: "10px" }}>EXPLAIN LIKE I'M 5</div>
                                <p style={{ margin: 0, color: "#cbd5e1", fontSize: "14px", lineHeight: "1.75", fontFamily: "'DM Sans', sans-serif" }}>{method.eli5}</p>
                            </div>
                            <div>
                                <div style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em", marginBottom: "8px" }}>TECHNICAL INTUITION</div>
                                <p style={{ margin: 0, color: "#94a3b8", fontSize: "13px", lineHeight: "1.7", fontFamily: "'DM Sans', sans-serif" }}>{method.intuition}</p>
                            </div>
                            {method.sensitivity && (
                                <div>
                                    <div style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em", marginBottom: "10px" }}>SENSITIVITY TO…</div>
                                    <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                                        <SensitivityPill label="Word Order" value={method.sensitivity.order} />
                                        <SensitivityPill label="Synonyms" value={method.sensitivity.synonyms} />
                                        <SensitivityPill label="Frequency" value={method.sensitivity.freq} />
                                        <SensitivityPill label="Typos" value={method.sensitivity.typos} />
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                    {tab === "formula" && (
                        <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: "16px" }}>
                            <div style={{ background: "#030c18", border: `1px solid ${colors.border}50`, borderRadius: "12px", padding: "20px" }}>
                                <div style={{ fontSize: "10px", color: colors.accent, fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em", marginBottom: "12px" }}>FORMULA</div>
                                <code style={{ color: colors.accent, fontFamily: "JetBrains Mono, monospace", fontSize: "14px", lineHeight: "1.8", display: "block", whiteSpace: "pre-wrap" }}>{method.formula}</code>
                            </div>
                            <div style={{ background: "#030c18", border: "1px solid #0f2236", borderRadius: "12px", padding: "16px" }}>
                                <div style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em", marginBottom: "10px" }}>WHAT EACH PART MEANS</div>
                                <p style={{ margin: 0, color: "#94a3b8", fontSize: "13px", lineHeight: "1.75", fontFamily: "'DM Sans', sans-serif" }}>{method.formulaExplained}</p>
                            </div>
                        </div>
                    )}
                    {tab === "example" && (
                        <div style={{ padding: "20px 24px" }}>
                            <div style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em", marginBottom: "12px" }}>STEP-BY-STEP CALCULATION</div>
                            <pre style={{ margin: 0, background: "#030c18", border: `1px solid ${colors.border}40`, borderRadius: "12px", padding: "18px", color: "#e2e8f0", fontFamily: "JetBrains Mono, monospace", fontSize: "12.5px", lineHeight: "1.9", whiteSpace: "pre-wrap", overflowX: "auto" }}>{method.example}</pre>
                        </div>
                    )}
                    {tab === "guide" && (
                        <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: "14px" }}>
                            <div style={{ background: "rgba(52,211,153,0.06)", border: "1px solid rgba(52,211,153,0.2)", borderRadius: "12px", padding: "16px" }}>
                                <div style={{ fontSize: "10px", color: "#34d399", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em", marginBottom: "10px" }}>✓ USE WHEN</div>
                                <p style={{ margin: 0, color: "#94a3b8", fontSize: "13px", lineHeight: "1.7", fontFamily: "'DM Sans', sans-serif" }}>{method.useWhen}</p>
                            </div>
                            <div style={{ background: "rgba(248,113,113,0.06)", border: "1px solid rgba(248,113,113,0.2)", borderRadius: "12px", padding: "16px" }}>
                                <div style={{ fontSize: "10px", color: "#f87171", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.12em", marginBottom: "10px" }}>✗ AVOID WHEN</div>
                                <p style={{ margin: 0, color: "#94a3b8", fontSize: "13px", lineHeight: "1.7", fontFamily: "'DM Sans', sans-serif" }}>{method.avoidWhen}</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}