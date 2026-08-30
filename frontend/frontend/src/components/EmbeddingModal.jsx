import { useState, useCallback, useEffect, useRef } from "react";
import { API_URL } from "../data/constants";

export default function EmbeddingModal({ text1, text2, modelId, ppOpts = {}, onClose }) {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [projMethod, setProjMethod] = useState("pca");
    const [dims, setDims] = useState(2);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [hovered, setHovered] = useState(null);

    const camera = useRef({ x: 0, y: 0, scale: 1 });
    const drag = useRef({ active: false, startX: 0, startY: 0, camX: 0, camY: 0 });
    const rot3d = useRef({ yaw: 0.4, pitch: -0.3 });
    const rotDrag = useRef({ active: false, startX: 0, startY: 0, yaw0: 0, pitch0: 0 });

    const COLOR1 = "#818cf8";
    const COLOR2 = "#34d399";
    const COLOR1_GLOW = "rgba(129,140,248,0.35)";
    const COLOR2_GLOW = "rgba(52,211,153,0.35)";

    const fetchData = useCallback(() => {
        setLoading(true); setError(null); setData(null);
        fetch(`${API_URL}/embeddings`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text1, text2, model_id: modelId, method: projMethod, dims, ...ppOpts }),
        })
            .then(r => r.json())
            .then(d => { if (d.error) throw new Error(d.error); setData(d); camera.current = { x: 0, y: 0, scale: 1 }; })
            .catch(e => setError(e.message))
            .finally(() => setLoading(false));
    }, [text1, text2, modelId, projMethod, dims]);

    useEffect(() => { fetchData(); }, [fetchData]);

    const rotate3d = (x, y, z, yaw, pitch) => {
        const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
        const x1 = cosY * x + sinY * z;
        const z1 = -sinY * x + cosY * z;
        const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
        const y2 = cosP * y - sinP * z1;
        const z2 = sinP * y + cosP * z1;
        return { sx: x1, sy: y2, sz: z2 };
    };

    const draw = useCallback(() => {
        if (!data || !canvasRef.current) return;
        const canvas = canvasRef.current;
        const W = canvas.width, H = canvas.height;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "#050d1a";
        ctx.fillRect(0, 0, W, H);

        ctx.strokeStyle = "rgba(99,102,241,0.06)";
        ctx.lineWidth = 1;
        const gridStep = 60;
        for (let gx = W / 2 % gridStep; gx < W; gx += gridStep) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke(); }
        for (let gy = H / 2 % gridStep; gy < H; gy += gridStep) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke(); }

        const { points } = data;
        const { x: camX, y: camY, scale } = camera.current;
        const cx = W / 2 + camX, cy = H / 2 + camY;

        const xs = points.map(p => p.x), ys = points.map(p => p.y);
        const zs = dims === 3 ? points.map(p => p.z) : points.map(() => 0);
        const xRange = Math.max(...xs) - Math.min(...xs) || 1;
        const yRange = Math.max(...ys) - Math.min(...ys) || 1;
        const zRange = dims === 3 ? (Math.max(...zs) - Math.min(...zs) || 1) : 1;
        const range = Math.max(xRange, yRange, zRange);
        const norm = points.map((p, i) => ({
            ...p,
            nx: (p.x - (Math.max(...xs) + Math.min(...xs)) / 2) / range,
            ny: (p.y - (Math.max(...ys) + Math.min(...ys)) / 2) / range,
            nz: dims === 3 ? (zs[i] - (Math.max(...zs) + Math.min(...zs)) / 2) / range : 0,
        }));

        const baseR = Math.min(W, H) * 0.38 * scale;
        let projected = norm.map(p => {
            if (dims === 3) {
                const { sx, sy, sz } = rotate3d(p.nx, p.ny, p.nz, rot3d.current.yaw, rot3d.current.pitch);
                const perspective = 1 / (1.8 + sz);
                return { ...p, sx: cx + sx * baseR * perspective, sy: cy + sy * baseR * perspective, depth: sz };
            }
            return { ...p, sx: cx + p.nx * baseR, sy: cy - p.ny * baseR, depth: 0 };
        });
        if (dims === 3) projected.sort((a, b) => a.depth - b.depth);

        const tok1Map = {}, tok2Map = {};
        projected.forEach(p => { if (p.source === 1) tok1Map[p.token] = p; else tok2Map[p.token] = p; });
        Object.keys(tok1Map).forEach(tok => {
            if (tok2Map[tok]) {
                const a = tok1Map[tok], b = tok2Map[tok];
                ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy);
                ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 0.8;
                ctx.setLineDash([3, 4]); ctx.stroke(); ctx.setLineDash([]);
            }
        });

        projected.forEach((p, i) => {
            const isHov = hovered && hovered.token === p.token && hovered.source === p.source && hovered._idx === i;
            const color = p.source === 1 ? COLOR1 : COLOR2;
            const glow = p.source === 1 ? COLOR1_GLOW : COLOR2_GLOW;
            const r = isHov ? 8 : 5;
            if (isHov) { ctx.beginPath(); ctx.arc(p.sx, p.sy, r + 6, 0, Math.PI * 2); ctx.fillStyle = glow; ctx.fill(); }
            ctx.beginPath(); ctx.arc(p.sx, p.sy, r + 1.5, 0, Math.PI * 2); ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fill();
            ctx.beginPath(); ctx.arc(p.sx, p.sy, r, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
            const showLabel = isHov || projected.length <= 20;
            if (showLabel) {
                ctx.font = isHov ? "bold 12px 'JetBrains Mono', monospace" : "10px 'JetBrains Mono', monospace";
                ctx.fillStyle = isHov ? "#f1f5f9" : color;
                ctx.textAlign = "left"; ctx.textBaseline = "middle";
                ctx.fillText(p.token, p.sx + r + 4, p.sy);
            }
        });
        canvas._projected = projected;
    }, [data, dims, hovered]);

    useEffect(() => { draw(); }, [draw]);

    useEffect(() => {
        if (!containerRef.current || !canvasRef.current) return;
        const obs = new ResizeObserver(() => {
            const el = containerRef.current;
            canvasRef.current.width = el.clientWidth;
            canvasRef.current.height = el.clientHeight;
            draw();
        });
        obs.observe(containerRef.current);
        return () => obs.disconnect();
    }, [draw]);

    const handleMouseMove = useCallback((e) => {
        const canvas = canvasRef.current;
        if (!canvas || !canvas._projected) return;
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
        const my = (e.clientY - rect.top) * (canvas.height / rect.height);
        let closest = null, minDist = 14;
        canvas._projected.forEach((p, i) => {
            const d = Math.hypot(p.sx - mx, p.sy - my);
            if (d < minDist) { minDist = d; closest = { ...p, _idx: i, screenX: e.clientX, screenY: e.clientY }; }
        });
        setHovered(closest);
    }, []);

    const handleMouseDown2D = (e) => { if (dims === 3) return; drag.current = { active: true, startX: e.clientX, startY: e.clientY, camX: camera.current.x, camY: camera.current.y }; };
    const handleMouseMove2D = (e) => { if (!drag.current.active || dims === 3) return; camera.current.x = drag.current.camX + e.clientX - drag.current.startX; camera.current.y = drag.current.camY + e.clientY - drag.current.startY; draw(); };
    const handleMouseUp2D = () => { drag.current.active = false; };
    const handleMouseDown3D = (e) => { if (dims !== 3) return; rotDrag.current = { active: true, startX: e.clientX, startY: e.clientY, yaw0: rot3d.current.yaw, pitch0: rot3d.current.pitch }; };
    const handleMouseMove3D = (e) => { if (!rotDrag.current.active || dims !== 3) return; rot3d.current.yaw = rotDrag.current.yaw0 + (e.clientX - rotDrag.current.startX) * 0.008; rot3d.current.pitch = rotDrag.current.pitch0 + (e.clientY - rotDrag.current.startY) * 0.008; draw(); };
    const handleMouseUp3D = () => { rotDrag.current.active = false; };
    const handleWheel = (e) => { e.preventDefault(); const factor = e.deltaY < 0 ? 1.1 : 0.9; camera.current.scale = Math.max(0.2, Math.min(8, camera.current.scale * factor)); draw(); };

    const combinedMouseDown = (e) => { handleMouseDown2D(e); handleMouseDown3D(e); };
    const combinedMouseMove = (e) => { handleMouseMove(e); handleMouseMove2D(e); handleMouseMove3D(e); };
    const combinedMouseUp = () => { handleMouseUp2D(); handleMouseUp3D(); };

    return (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)", backdropFilter: "blur(10px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1100, padding: "16px" }}
            onClick={onClose}>
            <div style={{ background: "#05101f", border: "1px solid #1e3a5f", borderRadius: "22px", width: "min(98vw, 1160px)", height: "min(92vh, 820px)", display: "flex", flexDirection: "column", boxShadow: "0 0 80px rgba(99,102,241,0.18)", overflow: "hidden" }}
                onClick={e => e.stopPropagation()}>

                <div style={{ padding: "18px 24px 14px", borderBottom: "1px solid #0f2744", display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexShrink: 0 }}>
                    <div>
                        <h2 style={{ margin: 0, fontFamily: "'DM Serif Display', serif", fontSize: "22px", color: "#f1f5f9" }}>Embedding Space Visualisation</h2>
                    </div>
                    <button onClick={onClose} style={{ background: "transparent", border: "1px solid #1e293b", color: "#475569", borderRadius: "8px", padding: "6px 10px", cursor: "pointer", fontSize: "16px", flexShrink: 0, marginLeft: "16px" }}>✕</button>
                </div>

                <div style={{ padding: "12px 24px", borderBottom: "1px solid #0f2744", display: "flex", alignItems: "center", gap: "20px", flexWrap: "wrap", flexShrink: 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em" }}>PROJECTION</span>
                        <div style={{ display: "flex", background: "#0a1628", border: "1px solid #1e293b", borderRadius: "8px", overflow: "hidden" }}>
                            {["pca", "tsne", "umap"].map(m => (
                                <button key={m} onClick={() => setProjMethod(m)}
                                    style={{ padding: "6px 14px", border: "none", background: projMethod === m ? "#6366f1" : "transparent", color: projMethod === m ? "white" : "#475569", fontFamily: "JetBrains Mono, monospace", fontSize: "11px", cursor: "pointer", fontWeight: projMethod === m ? 700 : 400, letterSpacing: "0.05em", transition: "all 0.15s" }}>
                                    {m.toUpperCase()}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em" }}>DIMS</span>
                        <div style={{ display: "flex", background: "#0a1628", border: "1px solid #1e293b", borderRadius: "8px", overflow: "hidden" }}>
                            {[2, 3].map(d => (
                                <button key={d} onClick={() => setDims(d)}
                                    style={{ padding: "6px 14px", border: "none", background: dims === d ? "#6366f1" : "transparent", color: dims === d ? "white" : "#475569", fontFamily: "JetBrains Mono, monospace", fontSize: "11px", cursor: "pointer", fontWeight: dims === d ? 700 : 400, transition: "all 0.15s" }}>
                                    {d}D
                                </button>
                            ))}
                        </div>
                    </div>
                    <span style={{ fontSize: "11px", background: "rgba(16,185,129,0.1)", border: "1px solid #10b981", color: "#34d399", borderRadius: "6px", padding: "3px 10px", fontFamily: "JetBrains Mono, monospace" }}>{modelId}</span>
                    <div style={{ display: "flex", gap: "16px", marginLeft: "auto" }}>
                        {[{ color: COLOR1, label: "Text 1" }, { color: COLOR2, label: "Text 2" }].map(({ color, label }) => (
                            <div key={label} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: color, display: "inline-block", boxShadow: `0 0 6px ${color}` }} />
                                <span style={{ fontSize: "11px", color: "#94a3b8", fontFamily: "JetBrains Mono, monospace" }}>{label}</span>
                            </div>
                        ))}
                    </div>
                </div>

                <div ref={containerRef} style={{ flex: 1, position: "relative", overflow: "hidden", cursor: dims === 3 ? "grab" : "crosshair" }}>
                    {loading && (
                        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "16px" }}>
                            <div style={{ width: "40px", height: "40px", border: "3px solid #0f2744", borderTopColor: "#6366f1", borderRadius: "50%", animation: "spin 0.9s linear infinite" }} />
                            <span style={{ color: "#334155", fontFamily: "JetBrains Mono, monospace", fontSize: "12px" }}>
                                {projMethod === "tsne" ? "Running t-SNE…" : projMethod === "umap" ? "Running UMAP…" : "Computing PCA…"}
                            </span>
                        </div>
                    )}
                    {error && (
                        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                            <div style={{ background: "#0f172a", border: "1px solid #ef4444", borderRadius: "12px", padding: "20px 28px", maxWidth: "480px" }}>
                                <div style={{ color: "#f87171", fontFamily: "JetBrains Mono, monospace", fontSize: "13px" }}>⚠ {error}</div>
                            </div>
                        </div>
                    )}
                    {!loading && !error && (
                        <canvas ref={canvasRef}
                            style={{ width: "100%", height: "100%", display: "block" }}
                            onMouseMove={combinedMouseMove} onMouseDown={combinedMouseDown}
                            onMouseUp={combinedMouseUp} onMouseLeave={combinedMouseUp} onWheel={handleWheel}
                        />
                    )}
                    {hovered && (
                        <div style={{ position: "fixed", left: hovered.screenX + 14, top: hovered.screenY - 8, background: "#0f172a", border: `1px solid ${hovered.source === 1 ? COLOR1 : COLOR2}`, borderRadius: "10px", padding: "10px 14px", pointerEvents: "none", zIndex: 1200, boxShadow: "0 4px 24px rgba(0,0,0,0.6)" }}>
                            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "15px", fontWeight: 700, color: hovered.source === 1 ? COLOR1 : COLOR2 }}>"{hovered.token}"</div>
                            <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: "11px", color: "#475569", marginTop: "4px" }}>from <strong style={{ color: "#94a3b8" }}>Text {hovered.source}</strong></div>
                            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "10px", color: "#1e3a5f", marginTop: "6px" }}>
                                x: {hovered.x.toFixed(3)}  y: {hovered.y.toFixed(3)}{dims === 3 ? `  z: ${hovered.z?.toFixed(3)}` : ""}
                            </div>
                        </div>
                    )}
                    {data && !loading && (
                        <div style={{ position: "absolute", bottom: "12px", right: "16px", fontSize: "10px", color: "#1e3a5f", fontFamily: "JetBrains Mono, monospace", userSelect: "none" }}>
                            {dims === 2 ? "drag to pan · scroll to zoom · hover for token" : "drag to rotate · scroll to zoom · hover for token"}
                        </div>
                    )}
                </div>

            </div>
        </div>
    );
}