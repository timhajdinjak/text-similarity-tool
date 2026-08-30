import { useState, useEffect, useRef } from "react";

export default function ModelSelector({ groups, selectedId, loadedId, onChange }) {
    const [open, setOpen] = useState(false);
    const ref = useRef(null);

    useEffect(() => {
        const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
        document.addEventListener("mousedown", handler);
        return () => document.removeEventListener("mousedown", handler);
    }, []);

    const selectedModel = groups.flatMap(g => g.models).find(m => m.id === selectedId);
    const isLoaded = selectedId === loadedId;

    return (
        <div ref={ref} style={{ position: "relative" }}>
            <div style={{ fontSize: "11px", fontFamily: "JetBrains Mono, monospace", color: "#6366f1", letterSpacing: "0.1em", marginBottom: "8px", display: "flex", alignItems: "center", gap: "8px" }}>
                EMBEDDING MODEL
                {loadedId && (
                    <span style={{ background: isLoaded ? "rgba(16,185,129,0.15)" : "rgba(245,158,11,0.15)", border: `1px solid ${isLoaded ? "#10b981" : "#f59e0b"}`, color: isLoaded ? "#34d399" : "#fbbf24", borderRadius: "4px", padding: "1px 7px", fontSize: "10px" }}>
                        {isLoaded ? `✓ loaded` : `⟳ will swap`}
                    </span>
                )}
            </div>

            <button
                onClick={() => setOpen(o => !o)}
                style={{ width: "100%", background: "#0f172a", border: "1px solid #1e293b", borderRadius: "12px", padding: "12px 16px", color: "#e2e8f0", fontFamily: "'DM Sans', sans-serif", fontSize: "14px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", textAlign: "left", transition: "border-color 0.2s" }}
                onMouseEnter={e => e.currentTarget.style.borderColor = "#6366f1"}
                onMouseLeave={e => e.currentTarget.style.borderColor = "#1e293b"}
            >
                <div>
                    <div style={{ fontWeight: 500 }}>{selectedModel?.label ?? selectedId}</div>
                    {selectedModel?.note && <div style={{ fontSize: "11px", color: "#475569", marginTop: "2px", fontFamily: "JetBrains Mono, monospace" }}>{selectedModel.note}</div>}
                </div>
                <span style={{ color: "#475569", transform: open ? "rotate(180deg)" : "none", transition: "transform 0.2s", marginLeft: "12px" }}>▼</span>
            </button>

            {open && (
                <div style={{ position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0, background: "#0d1526", border: "1px solid #1e293b", borderRadius: "12px", zIndex: 200, maxHeight: "380px", overflowY: "auto", boxShadow: "0 8px 40px rgba(0,0,0,0.6)" }}>
                    {groups.map(group => (
                        <div key={group.group}>
                            <div style={{ padding: "10px 16px 6px", fontSize: "10px", fontFamily: "JetBrains Mono, monospace", color: "#6366f1", letterSpacing: "0.1em", borderBottom: "1px solid #1e293b", background: "#080f1a", position: "sticky", top: 0 }}>
                                {group.group.toUpperCase()}
                            </div>
                            {group.models.map(model => {
                                const isActive = model.id === selectedId;
                                const isCurrentlyLoaded = model.id === loadedId;
                                return (
                                    <button
                                        key={model.id}
                                        onClick={() => { onChange(model.id); setOpen(false); }}
                                        style={{ width: "100%", background: isActive ? "rgba(99,102,241,0.1)" : "transparent", border: "none", padding: "10px 16px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", textAlign: "left", transition: "background 0.15s" }}
                                        onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}
                                        onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = "transparent"; }}
                                    >
                                        <div>
                                            <div style={{ color: isActive ? "#818cf8" : "#94a3b8", fontSize: "13px", fontFamily: "'DM Sans', sans-serif", fontWeight: isActive ? 600 : 400 }}>{model.label}</div>
                                            <div style={{ color: "#374151", fontSize: "11px", fontFamily: "JetBrains Mono, monospace", marginTop: "2px" }}>{model.note}</div>
                                        </div>
                                        <div style={{ display: "flex", gap: "6px", flexShrink: 0, marginLeft: "12px" }}>
                                            {isCurrentlyLoaded && <span style={{ fontSize: "10px", background: "rgba(16,185,129,0.15)", border: "1px solid #10b981", color: "#34d399", borderRadius: "4px", padding: "1px 6px" }}>in RAM</span>}
                                            {isActive && <span style={{ color: "#6366f1", fontSize: "14px" }}>✓</span>}
                                        </div>
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