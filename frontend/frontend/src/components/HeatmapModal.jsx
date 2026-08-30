import { useState, useEffect, useCallback, useRef } from "react";
import { API_URL } from "../data/constants";

export default function HeatmapModal({ text1, text2, modelId, methodName, methodKey, ppOpts = {}, onClose }) {
    const canvasRef = useRef(null);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [hovered, setHovered] = useState(null);
    const [tooltip, setTooltip] = useState({ x: 0, y: 0, visible: false });
    const [colorScheme, setColorScheme] = useState("plasma");

    const CELL = 34, LABEL_W = 90, LABEL_H = 90, PAD = 12;

    const colorScales = {
        plasma: (t) => interp(t, [[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]]),
        viridis: (t) => interp(t, [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]),
        cool: (t) => interp(t, [[10, 10, 50], [20, 60, 160], [0, 160, 220], [100, 220, 255], [230, 248, 255]]),
    };

    function interp(t, stops) {
        const i = t * (stops.length - 1);
        const lo = stops[Math.floor(i)], hi = stops[Math.min(Math.ceil(i), stops.length - 1)];
        const f = i - Math.floor(i);
        return `rgb(${Math.round(lo[0] + (hi[0] - lo[0]) * f)},${Math.round(lo[1] + (hi[1] - lo[1]) * f)},${Math.round(lo[2] + (hi[2] - lo[2]) * f)})`;
    }

    useEffect(() => {
        setLoading(true); setError(null);
        fetch(`${API_URL}/heatmap`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text1, text2, model_id: modelId, method_key: methodKey, ...ppOpts }),
        })
            .then(r => r.json())
            .then(d => { if (d.error) throw new Error(d.error); setData(d); })
            .catch(e => setError(e.message))
            .finally(() => setLoading(false));
    }, [text1, text2, modelId, methodKey]);

    useEffect(() => {
        if (!data || !canvasRef.current) return;
        const { tokens1, tokens2, matrix } = data;
        const rows = tokens1.length, cols = tokens2.length;
        const W = LABEL_W + cols * CELL + PAD;
        const H = LABEL_H + rows * CELL + PAD;
        const canvas = canvasRef.current;
        canvas.width = W; canvas.height = H;
        const ctx = canvas.getContext("2d");
        const scale = colorScales[colorScheme];

        ctx.fillStyle = "#080f1a";
        ctx.fillRect(0, 0, W, H);

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const val = matrix[r][c];
                ctx.fillStyle = scale(val);
                ctx.fillRect(LABEL_W + c * CELL, LABEL_H + r * CELL, CELL - 1, CELL - 1);
                if (CELL >= 30) {
                    ctx.fillStyle = val > 0.55 ? "rgba(0,0,0,0.75)" : "rgba(255,255,255,0.6)";
                    ctx.font = "bold 9px 'JetBrains Mono', monospace";
                    ctx.textAlign = "center"; ctx.textBaseline = "middle";
                    ctx.fillText(val.toFixed(2), LABEL_W + c * CELL + CELL / 2 - 0.5, LABEL_H + r * CELL + CELL / 2);
                }
            }
        }

        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        for (let r = 0; r < rows; r++) {
            ctx.fillStyle = "#94a3b8";
            const label = tokens1[r].length > 10 ? tokens1[r].slice(0, 9) + "…" : tokens1[r];
            ctx.fillText(label, LABEL_W - 8, LABEL_H + r * CELL + CELL / 2);
        }

        ctx.save();
        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        for (let c = 0; c < cols; c++) {
            ctx.save();
            ctx.translate(LABEL_W + c * CELL + CELL / 2, LABEL_H - 8);
            ctx.rotate(-Math.PI / 2.5);
            ctx.fillStyle = "#94a3b8";
            const label = tokens2[c].length > 10 ? tokens2[c].slice(0, 9) + "…" : tokens2[c];
            ctx.fillText(label, 0, 0);
            ctx.restore();
        }
        ctx.restore();

        ctx.font = "bold 10px 'JetBrains Mono', monospace";
        ctx.fillStyle = "#475569";
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillText("TEXT 2 TOKENS →", LABEL_W + (cols * CELL) / 2, 2);
        ctx.save();
        ctx.translate(6, LABEL_H + (rows * CELL) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = "top";
        ctx.fillText("TEXT 1 TOKENS →", 0, 0);
        ctx.restore();
    }, [data, colorScheme]);

    const handleMouseMove = useCallback((e) => {
        if (!data || !canvasRef.current) return;
        const rect = canvasRef.current.getBoundingClientRect();
        const scaleX = canvasRef.current.width / rect.width;
        const scaleY = canvasRef.current.height / rect.height;
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;
        const col = Math.floor((mx - LABEL_W) / CELL);
        const row = Math.floor((my - LABEL_H) / CELL);
        if (row >= 0 && row < data.tokens1.length && col >= 0 && col < data.tokens2.length) {
            setHovered({ row, col, value: data.matrix[row][col], t1: data.tokens1[row], t2: data.tokens2[col] });
            setTooltip({ x: e.clientX, y: e.clientY, visible: true });
        } else {
            setHovered(null);
            setTooltip(t => ({ ...t, visible: false }));
        }
    }, [data]);

    const handleMouseLeave = () => { setHovered(null); setTooltip(t => ({ ...t, visible: false })); };

    const minVal = data ? Math.min(...data.matrix.flat()) : 0;
    const maxVal = data ? Math.max(...data.matrix.flat()) : 1;
    const avgVal = data ? data.matrix.flat().reduce((a, b) => a + b, 0) / data.matrix.flat().length : 0;

    const gradientStops = {
        plasma: "rgb(13,8,135), rgb(126,3,168), rgb(204,71,120), rgb(248,149,64), rgb(240,249,33)",
        viridis: "rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37)",
        cool: "rgb(10,10,50), rgb(20,60,160), rgb(0,160,220), rgb(100,220,255), rgb(230,248,255)",
    };

    return (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.85)", backdropFilter: "blur(8px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1100, padding: "20px" }}
            onClick={onClose}>
            <div style={{ background: "#0a1628", border: "1px solid #1e3a5f", borderRadius: "20px", maxWidth: "min(95vw, 1100px)", width: "100%", maxHeight: "92vh", display: "flex", flexDirection: "column", boxShadow: "0 0 60px rgba(99,102,241,0.2)", overflow: "hidden" }}
                onClick={e => e.stopPropagation()}>

                <div style={{ padding: "20px 24px 16px", borderBottom: "1px solid #1e293b", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
                    <div>
                        <h2 style={{ margin: 0, fontFamily: "'DM Serif Display', serif", fontSize: "20px", color: "#f1f5f9" }}>{methodName}</h2>
                    </div>
                    <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                        {["plasma", "viridis", "cool"].map(s => (
                            <button key={s} onClick={() => setColorScheme(s)}
                                style={{ background: colorScheme === s ? "#1e3a5f" : "transparent", border: `1px solid ${colorScheme === s ? "#6366f1" : "#1e293b"}`, color: colorScheme === s ? "#818cf8" : "#475569", borderRadius: "8px", padding: "5px 12px", fontSize: "11px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer", letterSpacing: "0.05em" }}>
                                {s}
                            </button>
                        ))}
                        <button onClick={onClose} style={{ background: "transparent", border: "1px solid #374151", color: "#9ca3af", borderRadius: "8px", padding: "6px 10px", cursor: "pointer", fontSize: "16px", marginLeft: "4px" }}>✕</button>
                    </div>
                </div>

                {data && (
                    <div style={{ display: "flex", gap: "0", borderBottom: "1px solid #1e293b", flexShrink: 0 }}>
                        {[
                            { label: "TOKENS (T1 × T2)", value: `${data.tokens1.length} × ${data.tokens2.length}` },
                            { label: "MIN SIM", value: `${(minVal * 100).toFixed(1)}%` },
                            { label: "AVG SIM", value: `${(avgVal * 100).toFixed(1)}%` },
                            { label: "MAX SIM", value: `${(maxVal * 100).toFixed(1)}%` },
                            hovered && { label: "HOVERED", value: `"${hovered.t1}" ↔ "${hovered.t2}" = ${(hovered.value * 100).toFixed(1)}%` },
                        ].filter(Boolean).map((stat, i) => (
                            <div key={i} style={{ flex: 1, padding: "10px 16px", borderRight: i < 4 ? "1px solid #1e293b" : "none" }}>
                                <div style={{ fontSize: "9px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em", marginBottom: "3px" }}>{stat.label}</div>
                                <div style={{ fontSize: "13px", color: stat.label === "HOVERED" ? "#fbbf24" : "#e2e8f0", fontFamily: "JetBrains Mono, monospace", fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{stat.value}</div>
                            </div>
                        ))}
                    </div>
                )}

                <div style={{ flex: 1, overflow: "auto", padding: "20px", display: "flex", flexDirection: "column", gap: "16px" }}>
                    {loading && (
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", flex: 1, gap: "12px", color: "#475569", fontFamily: "JetBrains Mono, monospace", fontSize: "13px" }}>
                            <span style={{ width: "16px", height: "16px", border: "2px solid #1e293b", borderTopColor: "#6366f1", borderRadius: "50%", display: "inline-block", animation: "spin 0.8s linear infinite" }} />
                            Computing token embeddings…
                        </div>
                    )}
                    {error && <div style={{ color: "#f87171", fontFamily: "JetBrains Mono, monospace", fontSize: "13px", textAlign: "center" }}>⚠ {error}</div>}
                    {data && !loading && (
                        <>
                            <canvas ref={canvasRef} onMouseMove={handleMouseMove} onMouseLeave={handleMouseLeave}
                                style={{ maxWidth: "100%", borderRadius: "10px", cursor: "crosshair", imageRendering: "pixelated" }} />
                            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                                <span style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace" }}>0%</span>
                                <div style={{ flex: 1, height: "10px", borderRadius: "5px", background: `linear-gradient(to right, ${gradientStops[colorScheme]})` }} />
                                <span style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace" }}>100%</span>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {tooltip.visible && hovered && (
                <div style={{ position: "fixed", left: tooltip.x + 14, top: tooltip.y - 10, background: "#0f172a", border: "1px solid #6366f1", borderRadius: "8px", padding: "8px 12px", pointerEvents: "none", zIndex: 1200, boxShadow: "0 4px 20px rgba(0,0,0,0.5)" }}>
                    <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "12px", color: "#818cf8" }}>
                        <span style={{ color: "#fbbf24" }}>"{hovered.t1}"</span>
                        <span style={{ color: "#475569", margin: "0 6px" }}>↔</span>
                        <span style={{ color: "#34d399" }}>"{hovered.t2}"</span>
                    </div>
                    <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "14px", color: "#f1f5f9", fontWeight: 700, marginTop: "4px" }}>{(hovered.value * 100).toFixed(2)}% similarity</div>
                    <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: "10px", color: "#374151", marginTop: "2px" }}>row {hovered.row + 1} · col {hovered.col + 1}</div>
                </div>
            )}
        </div>
    );
}