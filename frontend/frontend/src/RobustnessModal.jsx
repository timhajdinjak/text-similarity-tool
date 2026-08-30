import { useState, useEffect, useRef, useCallback } from "react";

const API_URL = "http://localhost:5000";

const TYPE_COLORS = {
    "Token / Set-Based": { accent: "#818cf8", border: "#6366f1", bg: "rgba(99,102,241,0.15)" },
    "Vector / TF-IDF": { accent: "#34d399", border: "#10b981", bg: "rgba(16,185,129,0.15)" },
    "String / Character-Based": { accent: "#fbbf24", border: "#f59e0b", bg: "rgba(245,158,11,0.15)" },
    "Embedding / Semantic": { accent: "#f472b6", border: "#ec4899", bg: "rgba(236,72,153,0.15)" },
    "Token / N-Gram": { accent: "#60a5fa", border: "#3b82f6", bg: "rgba(59,130,246,0.15)" },
    "Sequence-Based": { accent: "#c084fc", border: "#a855f7", bg: "rgba(168,85,247,0.15)" },
    "NLG Evaluation": { accent: "#2dd4bf", border: "#14b8a6", bg: "rgba(20,184,166,0.15)" },
    "Information-Theoretic": { accent: "#f87171", border: "#ef4444", bg: "rgba(239,68,68,0.15)" },
};

const METHOD_META = {
    jaccard: { name: "Jaccard", type: "Token / Set-Based" },
    dice: { name: "Dice", type: "Token / Set-Based" },
    overlap: { name: "Overlap", type: "Token / Set-Based" },
    containment: { name: "Containment", type: "Token / Set-Based" },
    tfidf_cosine: { name: "TF-IDF Cosine", type: "Vector / TF-IDF" },
    levenshtein: { name: "Levenshtein", type: "String / Character-Based" },
    jaro_winkler: { name: "Jaro-Winkler", type: "String / Character-Based" },
    trigram: { name: "Char Trigram", type: "String / Character-Based" },
    bigram_overlap: { name: "Bigram Overlap", type: "Token / N-Gram" },
    lcs: { name: "LCS", type: "Sequence-Based" },
    hamming: { name: "Hamming", type: "Sequence-Based" },
    bleu: { name: "BLEU", type: "NLG Evaluation" },
    rouge_l: { name: "ROUGE-L", type: "NLG Evaluation" },
    rouge_2: { name: "ROUGE-2", type: "NLG Evaluation" },
    meteor: { name: "METEOR", type: "NLG Evaluation" },
    kl_divergence: { name: "KL Divergence", type: "Information-Theoretic" },
    jensen_shannon: { name: "Jensen-Shannon", type: "Information-Theoretic" },
    ncd: { name: "NCD", type: "Information-Theoretic" },
    embedding_cosine: { name: "Embed. Cosine", type: "Embedding / Semantic" },
    wmd: { name: "WMD", type: "Embedding / Semantic" },
    soft_cosine: { name: "Soft Cosine", type: "Embedding / Semantic" },
    bert_score: { name: "BERTScore", type: "Embedding / Semantic" },
};

const ALL_METHODS = Object.keys(METHOD_META);
const LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

function rng() { return Math.random(); }
function pick(arr) { return arr[Math.floor(rng() * arr.length)]; }
function randInt(lo, hi) { return Math.floor(rng() * (hi - lo + 1)) + lo; }

function applyTypos(text, rate) {
    const SUBS = {
        a: ["@", "4", "q", "s"], e: ["3", "€", "a", "r"], i: ["1", "!", "l", "j"],
        o: ["0", "oo", "p", "u"], l: ["1", "ll", "i"], s: ["$", "5", "z"],
        t: ["tt", "7", "y"], n: ["m", "nn", "h"], r: ["rr", "4", "t"],
        c: ["k", "cc", "v"], u: ["uu", "v", "y"], g: ["9", "gg", "q"],
        h: ["hh", "n", "j"], m: ["mm", "n", "w"], p: ["pp", "b", "o"],
        f: ["ff", "ph", "v"], b: ["bb", "p", "v"], d: ["dd", "t", "c"],
        k: ["kk", "c", "q"], w: ["ww", "v", "q"], y: ["yy", "i", "u"],
    };
    const DC = ["b", "d", "f", "g", "l", "m", "n", "p", "r", "s", "t", "z"];
    const extras = "aeiourstln";
    return text.split(" ").map(word => {
        if (word.length < 2 || rng() > rate) return word;
        const t = pick(["sub", "del", "double", "transpose", "insert"]);
        const chars = word.split("");
        const idx = randInt(0, chars.length - 1);
        const ch = chars[idx].toLowerCase();
        if (t === "sub" && SUBS[ch]) chars[idx] = pick(SUBS[ch]);
        else if (t === "del" && word.length > 2) chars.splice(idx, 1);
        else if (t === "double") chars.splice(randInt(0, chars.length - 1), 0, chars[randInt(0, chars.length - 1)]);
        else if (t === "transpose" && idx < chars.length - 1) [chars[idx], chars[idx + 1]] = [chars[idx + 1], chars[idx]];
        else if (t === "insert") chars.splice(idx, 0, pick(extras.split("")));
        return chars.join("");
    }).join(" ");
}

function applyShuffle(text, rate) {
    const words = text.split(" ");
    const n = Math.round(words.length * rate);
    for (let i = 0; i < n; i++) {
        const a = randInt(0, words.length - 1);
        const b = randInt(0, words.length - 1);
        [words[a], words[b]] = [words[b], words[a]];
    }
    return words.join(" ");
}

function applyNoise(text, rate) {
    const NOISE = ["um", "uh", "like", "er", "hmm", "xyz", "foo", "bar", "blah", "qux", "zzz", "etc", "...", "???", "###"];
    const CHARS = ["@", "#", "$", "%", "&", "*", "!", "~", "^", "_", "|", "<", ">"];
    return text.split(" ").flatMap(w => {
        const out = [w];
        if (rng() < rate) {
            const t = pick(["word", "char", "num"]);
            if (t === "word") out.push(pick(NOISE));
            else if (t === "char") out.push(pick(CHARS));
            else out.push(String(randInt(1, 999)));
        }
        return out;
    }).join(" ");
}

function corrupt(text, pct, type) {
    if (pct === 0) return text;
    const r = pct / 100;
    if (type === "typos") return applyTypos(text, r);
    if (type === "shuffle") return applyShuffle(text, r);
    if (type === "noise") return applyNoise(text, r);
    return text;
}

function SparkChart({ methodKey, levels, scores, corruptionType, isLoading }) {
    const canvasRef = useRef(null);
    const meta = METHOD_META[methodKey];
    const color = TYPE_COLORS[meta.type]?.accent || "#818cf8";
    const border = TYPE_COLORS[meta.type]?.border || "#6366f1";

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const W = canvas.width, H = canvas.height;
        const ctx = canvas.getContext("2d");
        const PAD = { t: 10, r: 10, b: 28, l: 32 };
        const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "#070e1b";
        ctx.fillRect(0, 0, W, H);

        ctx.strokeStyle = "rgba(255,255,255,0.04)";
        ctx.lineWidth = 1;
        [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
            const y = PAD.t + iH * (1 - v);
            ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + iW, y); ctx.stroke();
            ctx.fillStyle = "#1e3a5f";
            ctx.font = "8px JetBrains Mono, monospace";
            ctx.textAlign = "right";
            ctx.fillText((v * 100).toFixed(0) + "%", PAD.l - 3, y + 3);
        });

        ctx.fillStyle = "#334155";
        ctx.font = "8px JetBrains Mono, monospace";
        ctx.textAlign = "center";
        [0, 50, 100].forEach(pct => {
            const x = PAD.l + iW * (pct / 100);
            ctx.fillText(pct + "%", x, H - 6);
        });

        if (!scores || scores.length === 0) return;

        const pts = scores.map((s, i) => ({
            x: PAD.l + iW * (levels[i] / 100),
            y: PAD.t + iH * (1 - Math.max(0, Math.min(1, s))),
        }));

        ctx.beginPath();
        ctx.moveTo(pts[0].x, PAD.t + iH);
        pts.forEach(p => ctx.lineTo(p.x, p.y));
        ctx.lineTo(pts[pts.length - 1].x, PAD.t + iH);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, PAD.t, 0, PAD.t + iH);
        grad.addColorStop(0, color + "55");
        grad.addColorStop(1, color + "08");
        ctx.fillStyle = grad;
        ctx.fill();

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8;
        ctx.lineJoin = "round";
        pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
        ctx.stroke();

        pts.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        });

        const lastIdx = scores.length - 1;
        if (lastIdx >= 0) {
            ctx.font = "bold 9px JetBrains Mono, monospace";
            ctx.textAlign = "left";
            ctx.fillStyle = color;
            ctx.fillText((scores[0] * 100).toFixed(0) + "%", PAD.l + 2, PAD.t + iH * (1 - scores[0]) - 4);
            if (lastIdx > 0) {
                const last = scores[lastIdx];
                ctx.textAlign = "right";
                ctx.fillText((last * 100).toFixed(0) + "%", pts[lastIdx].x, pts[lastIdx].y - 4);
            }
        }
    }, [scores, levels, color]);

    const computedPct = scores ? scores.length : 0;
    const totalPct = levels.length;

    return (
        <div style={{
            background: "#07111f",
            border: `1px solid ${border}30`,
            borderRadius: "10px",
            padding: "10px",
            display: "flex",
            flexDirection: "column",
            gap: "6px",
            position: "relative",
            overflow: "hidden",
        }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color, letterSpacing: "0.04em", fontWeight: 600 }}>{meta.name}</span>
                <span style={{ fontSize: "9px", color: "#334155", fontFamily: "JetBrains Mono, monospace" }}>{meta.type.split(" / ")[0]}</span>
            </div>
            <canvas ref={canvasRef} width={220} height={110} style={{ width: "100%", borderRadius: "6px", imageRendering: "pixelated" }} />
            {isLoading && computedPct < totalPct && (
                <div style={{ position: "absolute", bottom: "10px", right: "10px", fontSize: "9px", color: "#334155", fontFamily: "JetBrains Mono, monospace" }}>
                    {computedPct}/{totalPct}
                </div>
            )}
        </div>
    );
}

function ThreeDView({ levels, allScores, visibleMethods, corruptionType }) {
    const canvasRef = useRef(null);
    const rotRef = useRef({ yaw: 0.5, pitch: 0.35 });
    const dragRef = useRef({ active: false, sx: 0, sy: 0, yaw0: 0, pitch0: 0 });
    const zoomRef = useRef(1.0);
    const [hovered, setHovered] = useState(null);
    const hitRef = useRef([]);

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const W = canvas.width, H = canvas.height;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, W, H);

        const bgGrad = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, W * 0.6);
        bgGrad.addColorStop(0, "#071020");
        bgGrad.addColorStop(1, "#030810");
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, W, H);

        const { yaw, pitch } = rotRef.current;
        const zoom = zoomRef.current;

        const rot = (x, y, z) => {
            const cy = Math.cos(yaw), sy2 = Math.sin(yaw);
            const x1 = cy * x + sy2 * z, z1 = -sy2 * x + cy * z;
            const cp = Math.cos(pitch), sp = Math.sin(pitch);
            return { x: x1, y: cp * y - sp * z1, z: sp * y + cp * z1 };
        };

        const originM = { x: -1, y: -1, z: -1 };
        const cx = W / 2, cy2 = H * 0.52;
        const scale = Math.min(W, H) * 0.28 * zoom;

        const proj = (mx, my, mz) => {
            const r = rot(mx, my, mz);
            const persp = 1 / (2.5 + r.z * 0.4);
            return { sx: cx + r.x * scale * persp, sy: cy2 - r.y * scale * persp, depth: r.z };
        };

        const p0 = proj(-1, -1, -1);
        const px = proj(1, -1, -1);
        const py = proj(-1, 1, -1);
        const pz = proj(-1, -1, 1);

        const drawAxis = (from, to, color, label, lx, ly) => {
            ctx.beginPath();
            ctx.moveTo(from.sx, from.sy);
            ctx.lineTo(to.sx, to.sy);
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.fillStyle = color;
            ctx.font = "bold 11px JetBrains Mono, monospace";
            ctx.textAlign = "center";
            ctx.fillText(label, to.sx + lx, to.sy + ly);
        };

        drawAxis(p0, px, "#475569", "Corruption %", 12, 12);
        drawAxis(p0, py, "#475569", "Similarity", -8, -8);
        drawAxis(p0, pz, "#475569", "Methods →", 14, 4);

        [0, 50, 100].forEach(pct => {
            const mx = (pct / 100) * 2 - 1;
            const tp = proj(mx, -1, -1);
            ctx.fillStyle = "#2a3f5a";
            ctx.font = "9px JetBrains Mono, monospace";
            ctx.textAlign = "center";
            ctx.fillText(pct + "%", tp.sx, tp.sy + 14);
        });

        [0, 0.5, 1.0].forEach(v => {
            const my = v * 2 - 1;
            const tp = proj(-1, my, -1);
            ctx.fillStyle = "#2a3f5a";
            ctx.font = "9px JetBrains Mono, monospace";
            ctx.textAlign = "right";
            ctx.fillText((v * 100).toFixed(0) + "%", tp.sx - 6, tp.sy + 3);
        });

        if (!allScores || visibleMethods.length === 0) return;

        const nM = visibleMethods.length;
        const lineObjects = [];

        visibleMethods.forEach((mkey, mi) => {
            const scores = allScores[mkey];
            if (!scores || scores.length === 0) return;
            const mz = nM === 1 ? 0 : (mi / (nM - 1)) * 2 - 1;
            const meta = METHOD_META[mkey];
            const color = TYPE_COLORS[meta.type]?.accent || "#818cf8";

            const pts = scores.map((s, i) => {
                const mx = (levels[i] / 100) * 2 - 1;
                const my = Math.max(0, Math.min(1, s)) * 2 - 1;
                return proj(mx, my, mz);
            });

            const avgDepth = pts.reduce((a, p) => a + p.depth, 0) / pts.length;
            lineObjects.push({ mkey, mi, mz, color, pts, avgDepth, scores, meta });
        });

        lineObjects.sort((a, b) => a.avgDepth - b.avgDepth);

        hitRef.current = [];

        lineObjects.forEach(({ mkey, color, pts, scores, meta }) => {
            if (pts.length < 2) return;

            ctx.beginPath();
            const floorPts = pts.map(p => {
                const fp = proj(
                    (levels[pts.indexOf(p)] / 100) * 2 - 1,
                    -1,
                    pts.indexOf(p) >= 0 ? (visibleMethods.indexOf(mkey) / Math.max(visibleMethods.length - 1, 1)) * 2 - 1 : 0
                );
                return fp;
            });
            ctx.moveTo(pts[0].sx, pts[0].sy);
            pts.forEach(p => ctx.lineTo(p.sx, p.sy));
            ctx.lineTo(pts[pts.length - 1].sx, pts[pts.length - 1].sy);
            floorPts.slice().reverse().forEach(p => ctx.lineTo(p.sx, p.sy));
            ctx.closePath();
            ctx.fillStyle = color + "18";
            ctx.fill();

            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.lineJoin = "round";
            pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.sx, p.sy) : ctx.lineTo(p.sx, p.sy));
            ctx.stroke();

            pts.forEach((p, i) => {
                const r = 3.5;
                ctx.beginPath();
                ctx.arc(p.sx, p.sy, r, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                hitRef.current.push({ sx: p.sx, sy: p.sy, r: 8, mkey, level: levels[i], score: scores[i], color, name: meta.name });
            });

            const last = pts[pts.length - 1];
            ctx.font = "bold 10px JetBrains Mono, monospace";
            ctx.fillStyle = color;
            ctx.textAlign = "left";
            ctx.fillText(meta.name, last.sx + 6, last.sy + 3);
        });
    }, [allScores, visibleMethods, levels]);

    useEffect(() => { draw(); }, [draw]);

    const containerRef = useRef(null);
    useEffect(() => {
        if (!containerRef.current || !canvasRef.current) return;
        const obs = new ResizeObserver(() => {
            const el = containerRef.current;
            if (!canvasRef.current) return;
            canvasRef.current.width = el.clientWidth;
            canvasRef.current.height = el.clientHeight;
            draw();
        });
        obs.observe(containerRef.current);
        return () => obs.disconnect();
    }, [draw]);

    const onMouseMove = useCallback((e) => {
        if (dragRef.current.active) {
            rotRef.current.yaw = dragRef.current.yaw0 + (e.clientX - dragRef.current.sx) * 0.009;
            rotRef.current.pitch = dragRef.current.pitch0 + (e.clientY - dragRef.current.sy) * 0.009;
            draw(); return;
        }
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
        const my = (e.clientY - rect.top) * (canvas.height / rect.height);
        let found = null;
        for (const h of hitRef.current) {
            if (Math.hypot(h.sx - mx, h.sy - my) < h.r) { found = { ...h, screenX: e.clientX, screenY: e.clientY }; break; }
        }
        setHovered(found);
    }, [draw]);

    const onWheel = useCallback((e) => {
        e.preventDefault();
        zoomRef.current = Math.max(0.3, Math.min(3, zoomRef.current * (e.deltaY < 0 ? 1.1 : 0.91)));
        draw();
    }, [draw]);

    return (
        <div ref={containerRef} style={{ flex: 1, position: "relative", cursor: "grab" }}>
            <canvas ref={canvasRef}
                style={{ width: "100%", height: "100%", display: "block" }}
                onMouseDown={e => { dragRef.current = { active: true, sx: e.clientX, sy: e.clientY, yaw0: rotRef.current.yaw, pitch0: rotRef.current.pitch }; }}
                onMouseMove={onMouseMove}
                onMouseUp={() => { dragRef.current.active = false; }}
                onMouseLeave={() => { dragRef.current.active = false; setHovered(null); }}
                onWheel={onWheel}
            />
            <div style={{ position: "absolute", bottom: "10px", right: "14px", fontSize: "10px", color: "#1e3a5f", fontFamily: "JetBrains Mono, monospace", userSelect: "none" }}>
                drag to rotate · scroll to zoom · hover dots for values
            </div>
            {hovered && (
                <div style={{ position: "fixed", left: hovered.screenX + 14, top: hovered.screenY - 10, background: "#0f172a", border: `1px solid ${hovered.color}`, borderRadius: "10px", padding: "10px 14px", pointerEvents: "none", zIndex: 1300, boxShadow: "0 4px 20px rgba(0,0,0,0.6)" }}>
                    <div style={{ fontSize: "13px", fontFamily: "JetBrains Mono, monospace", fontWeight: 700, color: hovered.color }}>{hovered.name}</div>
                    <div style={{ fontSize: "11px", color: "#64748b", marginTop: "4px", fontFamily: "JetBrains Mono, monospace" }}>
                        corruption: <span style={{ color: "#94a3b8" }}>{hovered.level}%</span>
                    </div>
                    <div style={{ fontSize: "15px", color: "#f1f5f9", fontFamily: "JetBrains Mono, monospace", fontWeight: 700, marginTop: "2px" }}>
                        {(hovered.score * 100).toFixed(1)}%
                    </div>
                </div>
            )}
        </div>
    );
}

export default function RobustnessModal({ text1, text2, modelId, ppOpts = {}, onClose }) {
    const [corruptionType, setCorruptionType] = useState("typos");
    const [viewMode, setViewMode] = useState("3d");
    const [allScores, setAllScores] = useState({});
    const [progress, setProgress] = useState(0);
    const [running, setRunning] = useState(false);
    const [done, setDone] = useState(false);
    const [error, setError] = useState(null);
    const cancelRef = useRef(false);

    const [visibleMethods, setVisibleMethods] = useState(ALL_METHODS);
    const toggleMethod = (k) => setVisibleMethods(v => v.includes(k) ? (v.length > 1 ? v.filter(x => x !== k) : v) : [...v, k]);

    const run = useCallback(async () => {
        if (!text1.trim() || !text2.trim()) return;
        cancelRef.current = false;
        setRunning(true);
        setDone(false);
        setError(null);
        setAllScores({});
        setProgress(0);

        const total = LEVELS.length;
        const scoreMap = {};

        for (let li = 0; li < LEVELS.length; li++) {
            if (cancelRef.current) break;
            const pct = LEVELS[li];
            const corrupted = corrupt(text2, pct, corruptionType);

            try {
                const res = await fetch(`${API_URL}/similarity`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text1, text2: corrupted, model_id: modelId, ...ppOpts }),
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                ALL_METHODS.forEach(k => {
                    if (!scoreMap[k]) scoreMap[k] = [];
                    scoreMap[k].push(data.results[k]?.score ?? 0);
                });
            } catch (e) {
                setError(e.message);
                break;
            }

            setAllScores(JSON.parse(JSON.stringify(scoreMap)));
            setProgress(Math.round(((li + 1) / total) * 100));
        }

        setRunning(false);
        setDone(true);
    }, [text1, text2, modelId, ppOpts, corruptionType]);

    useEffect(() => { run(); return () => { cancelRef.current = true; }; }, [run]);

    const robustnessScores = ALL_METHODS.map(k => {
        const s = allScores[k];
        if (!s || s.length < 2) return { key: k, auc: null };
        const auc = s.slice(1).reduce((sum, v, i) => sum + (s[i] + v) / 2 * (LEVELS[i + 1] - LEVELS[i]) / 100, 0) / 1;
        return { key: k, auc: auc / s[0] };
    }).filter(x => x.auc !== null).sort((a, b) => b.auc - a.auc);

    const completedLevels = allScores[ALL_METHODS[0]]?.length ?? 0;
    const displayLevels = LEVELS.slice(0, completedLevels);

    const TYPE_ORDER = Object.keys(TYPE_COLORS);
    const methodsByType = TYPE_ORDER.reduce((acc, t) => {
        acc[t] = ALL_METHODS.filter(k => METHOD_META[k].type === t);
        return acc;
    }, {});

    return (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)", backdropFilter: "blur(10px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1200, padding: "16px" }}
            onClick={onClose}>
            <div style={{ background: "#05101f", border: "1px solid #1e3a5f", borderRadius: "22px", width: "min(98vw, 1280px)", height: "min(94vh, 900px)", display: "flex", flexDirection: "column", boxShadow: "0 0 80px rgba(99,102,241,0.18)", overflow: "hidden" }}
                onClick={e => e.stopPropagation()}>

                <div style={{ padding: "18px 24px 14px", borderBottom: "1px solid #0f2744", display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexShrink: 0, gap: "16px" }}>
                    <div>
                        <h2 style={{ margin: 0, fontFamily: "'DM Serif Display', serif", fontSize: "22px", color: "#f1f5f9" }}>Robustness Curve</h2>
                    </div>
                    <button onClick={onClose} style={{ background: "transparent", border: "1px solid #1e293b", color: "#475569", borderRadius: "8px", padding: "6px 10px", cursor: "pointer", fontSize: "16px", flexShrink: 0 }}>✕</button>
                </div>

                <div style={{ padding: "12px 24px", borderBottom: "1px solid #0f2744", display: "flex", alignItems: "center", gap: "20px", flexWrap: "wrap", flexShrink: 0 }}>

                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em" }}>CORRUPTION</span>
                        <div style={{ display: "flex", background: "#0a1628", border: "1px solid #1e293b", borderRadius: "8px", overflow: "hidden" }}>
                            {[
                                { id: "typos", label: "✗ Typos" },
                                { id: "shuffle", label: "⇄ Shuffle" },
                                { id: "noise", label: "~ Noise" },
                            ].map(c => (
                                <button key={c.id} onClick={() => { if (!running) setCorruptionType(c.id); }}
                                    disabled={running}
                                    style={{ padding: "6px 14px", border: "none", background: corruptionType === c.id ? "#6366f1" : "transparent", color: corruptionType === c.id ? "white" : running ? "#1e293b" : "#475569", fontFamily: "JetBrains Mono, monospace", fontSize: "11px", cursor: running ? "not-allowed" : "pointer", fontWeight: corruptionType === c.id ? 700 : 400, letterSpacing: "0.05em", transition: "all 0.15s" }}>
                                    {c.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ fontSize: "10px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em" }}>VIEW</span>
                        <div style={{ display: "flex", background: "#0a1628", border: "1px solid #1e293b", borderRadius: "8px", overflow: "hidden" }}>
                            {[{ id: "3d", label: "3D" }, { id: "2d", label: "2D Grid" }].map(v => (
                                <button key={v.id} onClick={() => setViewMode(v.id)}
                                    style={{ padding: "6px 16px", border: "none", background: viewMode === v.id ? "#6366f1" : "transparent", color: viewMode === v.id ? "white" : "#475569", fontFamily: "JetBrains Mono, monospace", fontSize: "11px", cursor: "pointer", fontWeight: viewMode === v.id ? 700 : 400, transition: "all 0.15s" }}>
                                    {v.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    <span style={{ fontSize: "11px", background: "rgba(16,185,129,0.1)", border: "1px solid #10b981", color: "#34d399", borderRadius: "6px", padding: "3px 10px", fontFamily: "JetBrains Mono, monospace" }}>{modelId}</span>

                    {!running && done && (
                        <button onClick={run}
                            style={{ display: "flex", alignItems: "center", gap: "6px", background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.4)", color: "#818cf8", borderRadius: "8px", padding: "6px 14px", fontSize: "11px", fontFamily: "JetBrains Mono, monospace", cursor: "pointer" }}>
                            ↺ Re-run
                        </button>
                    )}

                    <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "10px" }}>
                        {running && (
                            <>
                                <div style={{ width: "120px", height: "4px", background: "#0f2744", borderRadius: "2px", overflow: "hidden" }}>
                                    <div style={{ width: `${progress}%`, height: "100%", background: "linear-gradient(90deg, #6366f1, #8b5cf6)", borderRadius: "2px", transition: "width 0.3s ease" }} />
                                </div>
                                <span style={{ fontSize: "11px", color: "#475569", fontFamily: "JetBrains Mono, monospace" }}>
                                    {completedLevels}/{LEVELS.length} levels
                                </span>
                            </>
                        )}
                        {done && !running && (
                            <span style={{ fontSize: "11px", color: "#34d399", fontFamily: "JetBrains Mono, monospace" }}>✓ complete</span>
                        )}
                    </div>
                </div>

                <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

                    {viewMode === "3d" && (
                        <>
                            <div style={{ width: "200px", borderRight: "1px solid #0f2744", display: "flex", flexDirection: "column", flexShrink: 0, overflowY: "auto" }}>
                                <div style={{ padding: "12px 14px", borderBottom: "1px solid #0f2744" }}>
                                    <div style={{ fontSize: "9px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em", marginBottom: "8px" }}>TOGGLE METHODS</div>
                                    {TYPE_ORDER.map(type => {
                                        const c = TYPE_COLORS[type];
                                        const keys = methodsByType[type];
                                        if (!keys || keys.length === 0) return null;
                                        return (
                                            <div key={type} style={{ marginBottom: "8px" }}>
                                                <div style={{ fontSize: "8px", color: c.accent, fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.08em", marginBottom: "4px", paddingLeft: "4px", borderLeft: `2px solid ${c.border}` }}>
                                                    {type.split(" / ")[0].toUpperCase()}
                                                </div>
                                                {keys.map(k => (
                                                    <button key={k} onClick={() => toggleMethod(k)}
                                                        style={{ display: "flex", alignItems: "center", gap: "6px", width: "100%", padding: "3px 6px", borderRadius: "5px", border: "none", background: visibleMethods.includes(k) ? c.bg : "transparent", cursor: "pointer", transition: "all 0.15s", marginBottom: "1px" }}>
                                                        <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: visibleMethods.includes(k) ? c.accent : "#1e293b", flexShrink: 0, transition: "background 0.15s" }} />
                                                        <span style={{ fontSize: "10px", fontFamily: "JetBrains Mono, monospace", color: visibleMethods.includes(k) ? c.accent : "#334155", textAlign: "left" }}>{METHOD_META[k].name}</span>
                                                    </button>
                                                ))}
                                            </div>
                                        );
                                    })}
                                </div>

                                {robustnessScores.length > 0 && (
                                    <div style={{ padding: "12px 14px" }}>
                                        <div style={{ fontSize: "9px", color: "#475569", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.1em", marginBottom: "8px" }}>
                                            ROBUSTNESS RANK
                                            <div style={{ fontSize: "8px", color: "#1e3a5f", marginTop: "2px" }}>(AUC / baseline)</div>
                                        </div>
                                        {robustnessScores.slice(0, 22).map((r, i) => {
                                            const color = TYPE_COLORS[METHOD_META[r.key].type]?.accent || "#818cf8";
                                            return (
                                                <div key={r.key} style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "5px" }}>
                                                    <span style={{ fontSize: "9px", color: "#1e3a5f", fontFamily: "JetBrains Mono, monospace", width: "14px", textAlign: "right", flexShrink: 0 }}>{i + 1}</span>
                                                    <span style={{ fontSize: "9px", fontFamily: "JetBrains Mono, monospace", color, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{METHOD_META[r.key].name}</span>
                                                    <span style={{ fontSize: "9px", fontFamily: "JetBrains Mono, monospace", color: r.auc > 0.8 ? "#34d399" : r.auc > 0.5 ? "#fbbf24" : "#f87171", flexShrink: 0 }}>{(r.auc * 100).toFixed(0)}%</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>

                            <ThreeDView levels={displayLevels} allScores={allScores} visibleMethods={visibleMethods.filter(k => allScores[k]?.length > 0)} corruptionType={corruptionType} />
                        </>
                    )}

                    {viewMode === "2d" && (
                        <div style={{ flex: 1, overflowY: "auto", padding: "20px 24px" }}>
                            {error && <div style={{ color: "#f87171", fontFamily: "JetBrains Mono, monospace", fontSize: "13px", marginBottom: "16px" }}>⚠ {error}</div>}
                            {TYPE_ORDER.map(type => {
                                const c = TYPE_COLORS[type];
                                const keys = methodsByType[type];
                                if (!keys || keys.length === 0) return null;
                                return (
                                    <div key={type} style={{ marginBottom: "28px" }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
                                            <span style={{ width: "3px", height: "18px", background: c.border, borderRadius: "2px", display: "inline-block" }} />
                                            <span style={{ fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color: c.accent, letterSpacing: "0.1em" }}>{type.toUpperCase()}</span>
                                        </div>
                                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "10px" }}>
                                            {keys.map(k => (
                                                <SparkChart
                                                    key={k}
                                                    methodKey={k}
                                                    levels={displayLevels}
                                                    scores={allScores[k]}
                                                    corruptionType={corruptionType}
                                                    isLoading={running}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}