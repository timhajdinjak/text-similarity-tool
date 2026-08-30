import { useState } from "react";
import { TYPE_COLORS } from "../data/constants";
import { highlightText } from "../utils.jsx";
import { ScoreRing } from "./UIComponents";
import HeatmapModal from "./HeatmapModal";

export default function SimilarityCard({ method, text1, text2, modelId, ppOpts = {}, onInfo }) {
    const [expanded, setExpanded] = useState(false);
    const [showHeatmap, setShowHeatmap] = useState(false);
    const colors = TYPE_COLORS[method.type] || { bg: "rgba(99,102,241,0.15)", border: "#6366f1", accent: "#818cf8" };
    const hasData = method.score !== null && text1 && text2;

    return (
        <>
            <div style={{ background: "#0f172a", border: `1px solid ${expanded ? colors.border : "#1e293b"}`, borderRadius: "14px", overflow: "hidden", transition: "border-color 0.2s, box-shadow 0.2s", boxShadow: expanded ? `0 0 20px ${colors.border}30` : "none", height: "fit-content" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "16px 18px", cursor: "pointer", userSelect: "none" }}
                    onClick={() => setExpanded(e => !e)}>
                    <ScoreRing score={method.score} color={colors.accent} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: "13px", fontFamily: "JetBrains Mono, monospace", color: colors.accent, letterSpacing: "0.05em", marginBottom: "3px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{method.type}</div>
                        <div style={{ fontSize: "15px", fontFamily: "'DM Serif Display', serif", color: "#e2e8f0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{method.name}</div>
                    </div>
                    <div style={{ display: "flex", gap: "8px", alignItems: "center", flexShrink: 0 }}>
                        <button onClick={e => { e.stopPropagation(); onInfo(method); }}
                            style={{ background: "#1e293b", border: "1px solid #334155", color: "#64748b", borderRadius: "6px", width: "28px", height: "28px", cursor: "pointer", fontSize: "13px", display: "flex", alignItems: "center", justifyContent: "center" }}
                            title="Method info">ⓘ</button>
                        <span style={{ color: "#475569", fontSize: "12px", transform: expanded ? "rotate(180deg)" : "rotate(0)", transition: "transform 0.2s" }}>▼</span>
                    </div>
                </div>

                {expanded && (
                    <div style={{ borderTop: `1px solid ${colors.border}30`, padding: "18px", background: colors.bg }}>
                        {!hasData ? (
                            <p style={{ color: "#475569", fontFamily: "'DM Sans', sans-serif", fontSize: "13px", textAlign: "center", margin: 0 }}>Enter both texts above to see what drives the score.</p>
                        ) : (
                            <>
                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                                    {[{ label: "Text 1", text: text1, highlights: method.highlights_text1 }, { label: "Text 2", text: text2, highlights: method.highlights_text2 }].map(({ label, text, highlights }) => (
                                        <div key={label} style={{ background: "#080f1a", borderRadius: "10px", padding: "14px", border: "1px solid #1e293b" }}>
                                            <div style={{ fontSize: "10px", color: colors.accent, fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em", marginBottom: "8px" }}>{label} - highlighted contributions</div>
                                            <p style={{ margin: 0, color: "#94a3b8", fontSize: "13px", lineHeight: "1.7", fontFamily: "'DM Sans', sans-serif" }}>{highlightText(text, highlights)}</p>
                                            {highlights.length > 0 && (
                                                <div style={{ marginTop: "10px", display: "flex", flexWrap: "wrap", gap: "4px" }}>
                                                    {[...new Set(highlights)].slice(0, 8).map(w => (
                                                        <span key={w} style={{ background: "rgba(251,191,36,0.15)", border: "1px solid rgba(251,191,36,0.3)", color: "#fcd34d", borderRadius: "4px", padding: "2px 7px", fontSize: "11px", fontFamily: "JetBrains Mono, monospace" }}>{w}</span>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>

                                <div style={{ marginTop: "14px", display: "flex", justifyContent: "center" }}>
                                    <button
                                        onClick={() => setShowHeatmap(true)}
                                        style={{ background: "linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15))", border: "1px solid rgba(99,102,241,0.4)", color: "#818cf8", borderRadius: "10px", padding: "9px 22px", fontSize: "12px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", letterSpacing: "0.06em", display: "flex", alignItems: "center", gap: "8px", transition: "all 0.2s" }}
                                        onMouseEnter={e => { e.currentTarget.style.background = "linear-gradient(135deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3))"; e.currentTarget.style.borderColor = "#6366f1"; }}
                                        onMouseLeave={e => { e.currentTarget.style.background = "linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15))"; e.currentTarget.style.borderColor = "rgba(99,102,241,0.4)"; }}
                                    >
                                        Show Token Heatmap
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                )}
            </div>

            {showHeatmap && (
                <HeatmapModal text1={text1} text2={text2} modelId={modelId} ppOpts={ppOpts} methodName={method.name} methodKey={method.key} onClose={() => setShowHeatmap(false)} />
            )}
        </>
    );
}