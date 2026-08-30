export function highlightText(text, words) {
    if (!words || words.length === 0) return <span>{text}</span>;
    const set = new Set(words.map(w => w.toLowerCase()));
    return text.split(/(\s+)/).map((tok, i) => {
        const clean = tok.replace(/[^a-z0-9]/gi, "").toLowerCase();
        return set.has(clean)
            ? <mark key={i} style={{ background: "rgba(251,191,36,0.3)", color: "#fcd34d", borderRadius: "3px", padding: "0 2px" }}>{tok}</mark>
            : <span key={i}>{tok}</span>;
    });
}