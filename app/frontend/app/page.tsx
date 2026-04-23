"use client";

import { useState } from "react";

const MAPS = [
  "Bind", "Haven", "Split", "Ascent", "Icebox",
  "Breeze", "Fracture", "Pearl", "Lotus", "Sunset"
];

type PredictResult = {
  killRange: [number, number];
  overProbability: number;
  archetype: string;
  similarPlayers: string[];
  model: string;
};

export default function Home() {
  const [player, setPlayer] = useState("");
  const [map, setMap] = useState(MAPS[0]);
  const [killLine, setKillLine] = useState("15.5");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handlePredict() {
    if (!player.trim()) { setError("Enter a player name."); return; }
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player: player.trim(), map, killLine: parseFloat(killLine) }),
      });
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Prediction failed. Make sure the API is running.");
    } finally {
      setLoading(false);
    }
  }

  const overPct = result ? Math.round(result.overProbability * 100) : 0;

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white font-mono relative overflow-hidden">

      {/* Background grid */}
      <div className="pointer-events-none fixed inset-0"
        style={{
          backgroundImage: `linear-gradient(rgba(255,70,85,0.04) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(255,70,85,0.04) 1px, transparent 1px)`,
          backgroundSize: "40px 40px"
        }} />

      {/* Top bar */}
      <div className="border-b border-[#FF4655]/20 px-8 py-4 flex items-center justify-between relative z-10">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-[#FF4655] animate-pulse" />
          <span className="text-[#FF4655] text-xs tracking-[0.3em] font-bold uppercase">
            Valorant Kill Predictor
          </span>
        </div>
        <span className="text-[#444] text-xs tracking-widest">CSCI-UA 473 · NYU · 2026</span>
      </div>

      <div className="max-w-3xl mx-auto px-6 py-16 relative z-10">

        {/* Hero */}
        <div className="mb-14">
          <p className="text-[#FF4655] text-xs tracking-[0.4em] mb-3 uppercase">VCT 2025–2026 · 1,707 Matches · 41K+ Map Stats</p>
          <h1 className="text-5xl font-black leading-none tracking-tight mb-4">
            PREDICT<br />
            <span className="text-[#FF4655]">KILL LINES</span>
          </h1>
          <p className="text-[#555] text-sm leading-relaxed max-w-md">
            Enter a pro player and map. Our model outputs a predicted kill range
            and over/under probability using quantile regression trained on VCT match data.
          </p>
        </div>

        {/* Input card */}
        <div className="border border-[#1f1f1f] bg-[#0f0f0f] p-8 mb-6">
          <div className="grid grid-cols-1 gap-5 mb-6">

            {/* Player */}
            <div>
              <label className="text-[#FF4655] text-xs tracking-[0.3em] uppercase mb-2 block">
                Player
              </label>
              <input
                type="text"
                value={player}
                onChange={e => setPlayer(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handlePredict()}
                placeholder="e.g. TenZ, aspas, Derke"
                className="w-full bg-[#141414] border border-[#2a2a2a] text-white px-4 py-3 text-sm
                           focus:outline-none focus:border-[#FF4655] transition-colors placeholder-[#333]"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Map */}
              <div>
                <label className="text-[#FF4655] text-xs tracking-[0.3em] uppercase mb-2 block">
                  Map
                </label>
                <select
                  value={map}
                  onChange={e => setMap(e.target.value)}
                  className="w-full bg-[#141414] border border-[#2a2a2a] text-white px-4 py-3 text-sm
                             focus:outline-none focus:border-[#FF4655] transition-colors appearance-none cursor-pointer"
                >
                  {MAPS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              {/* Kill line */}
              <div>
                <label className="text-[#FF4655] text-xs tracking-[0.3em] uppercase mb-2 block">
                  Kill Line
                </label>
                <input
                  type="number"
                  value={killLine}
                  onChange={e => setKillLine(e.target.value)}
                  step="0.5"
                  min="0"
                  className="w-full bg-[#141414] border border-[#2a2a2a] text-white px-4 py-3 text-sm
                             focus:outline-none focus:border-[#FF4655] transition-colors"
                />
              </div>
            </div>
          </div>

          {error && <p className="text-[#FF4655] text-xs mb-4">{error}</p>}

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full bg-[#FF4655] hover:bg-[#e03545] disabled:bg-[#3a1a1d] disabled:text-[#666]
                       text-white font-black text-sm tracking-[0.2em] uppercase py-4
                       transition-colors duration-150"
          >
            {loading ? "RUNNING MODEL..." : "PREDICT"}
          </button>
        </div>

        {/* Results */}
        {result && (
          <div className="border border-[#FF4655]/30 bg-[#0f0f0f] p-8 animate-in fade-in duration-300">

            {/* Header */}
            <div className="flex items-start justify-between mb-8 border-b border-[#1f1f1f] pb-6">
              <div>
                <p className="text-[#555] text-xs tracking-widest uppercase mb-1">Prediction for</p>
                <h2 className="text-2xl font-black tracking-tight">{player.toUpperCase()}</h2>
                <p className="text-[#444] text-xs mt-1">{map} · Line {killLine}</p>
              </div>
              <div className="text-right">
                <span className="inline-block border border-[#FF4655]/40 text-[#FF4655] text-xs
                                 tracking-widest uppercase px-3 py-1">
                  {result.archetype}
                </span>
                <p className="text-[#333] text-xs mt-2">{result.model}</p>
              </div>
            </div>

            {/* Kill range */}
            <div className="mb-8">
              <p className="text-[#555] text-xs tracking-[0.3em] uppercase mb-3">Predicted Kill Range</p>
              <div className="flex items-end gap-3">
                <span className="text-6xl font-black text-white leading-none">{result.killRange[0]}</span>
                <span className="text-[#333] text-2xl font-black mb-2">—</span>
                <span className="text-6xl font-black text-[#FF4655] leading-none">{result.killRange[1]}</span>
                <span className="text-[#444] text-sm mb-3 ml-1">kills</span>
              </div>
            </div>

            {/* Over/under bar */}
            <div className="mb-8">
              <div className="flex justify-between text-xs tracking-widest uppercase mb-2">
                <span className="text-[#555]">Under {killLine}</span>
                <span className="text-[#555]">Over {killLine}</span>
              </div>
              <div className="h-2 bg-[#1a1a1a] w-full relative">
                <div
                  className="h-full bg-[#FF4655] transition-all duration-700"
                  style={{ width: `${overPct}%` }}
                />
                <div
                  className="absolute top-1/2 -translate-y-1/2 w-px h-4 bg-[#444]"
                  style={{ left: "50%" }}
                />
              </div>
              <div className="flex justify-between mt-2">
                <span className="text-xs text-[#444]">{100 - overPct}%</span>
                <span className="text-xs font-bold text-[#FF4655]">{overPct}% OVER</span>
              </div>
            </div>

            {/* Similar players */}
            <div>
              <p className="text-[#555] text-xs tracking-[0.3em] uppercase mb-3">Similar Players via KNN</p>
              <div className="flex gap-3">
                {result.similarPlayers.map((p, i) => (
                  <div key={i} className="border border-[#1f1f1f] px-4 py-2 text-sm text-[#888] hover:border-[#FF4655]/40 hover:text-white transition-colors cursor-pointer">
                    {p}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
