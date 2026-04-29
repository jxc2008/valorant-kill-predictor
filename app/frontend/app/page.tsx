"use client";

import { useState } from "react";

const MAPS = [
  "Bind", "Haven", "Split", "Ascent", "Icebox",
  "Breeze", "Fracture", "Pearl", "Lotus", "Sunset"
];

type FeatureCoefficient = {
  feature: string;
  weight: number;
};

type PredictResult = {
  killRange: [number, number];
  overProbability: number;
  archetype: string;
  similarPlayers: string[];
  model: string;
  featureCoefficients?: FeatureCoefficient[];
};

type ModelType = "mlp" | "quantile_regression";

export default function Home() {
  const [player, setPlayer] = useState("");
  const [map, setMap] = useState(MAPS[0]);
  const [killLine, setKillLine] = useState("15.5");
  const [model, setModel] = useState<ModelType>("mlp");
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
        body: JSON.stringify({
          player: player.trim(),
          map,
          killLine: parseFloat(killLine),
          model
        }),
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
  const maxWeight = result?.featureCoefficients
    ? Math.max(...result.featureCoefficients.map(f => f.weight))
    : 1;

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
          <p className="text-[#FF4655] text-xs tracking-[0.4em] mb-3 uppercase">
            VCT 2025-2026 · 1,707 Matches · 41K+ Map Stats
          </p>
          <h1 className="text-5xl font-black leading-none tracking-tight mb-4">
            PREDICT<br />
            <span className="text-[#FF4655]">KILL LINES</span>
          </h1>
          <p className="text-[#555] text-sm leading-relaxed max-w-md">
            Enter a pro player and map. Our model outputs a predicted kill range
            and over/under probability using match data from VCT 2025-2026.
          </p>
        </div>

        {/* Model toggle */}
        <div className="mb-6">
          <p className="text-[#FF4655] text-xs tracking-[0.3em] uppercase mb-3">Model</p>
          <div className="flex gap-0 border border-[#2a2a2a] w-fit">
            <button
              onClick={() => { setModel("mlp"); setResult(null); }}
              className={`px-6 py-2.5 text-xs tracking-[0.2em] uppercase font-bold transition-colors
                ${model === "mlp"
                  ? "bg-[#FF4655] text-white"
                  : "bg-[#0f0f0f] text-[#555] hover:text-white"}`}
            >
              MLP Neural Net
            </button>
            <button
              onClick={() => { setModel("quantile_regression"); setResult(null); }}
              className={`px-6 py-2.5 text-xs tracking-[0.2em] uppercase font-bold transition-colors border-l border-[#2a2a2a]
                ${model === "quantile_regression"
                  ? "bg-[#FF4655] text-white"
                  : "bg-[#0f0f0f] text-[#555] hover:text-white"}`}
            >
              Quantile Regression
            </button>
          </div>
          <p className="text-[#333] text-xs mt-2">
            {model === "mlp"
              ? "Default · 2-layer neural network trained with pinball loss"
              : "Interpretable · linear model with per-feature coefficients by quantile"}
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

            {/* Kill distribution bar */}
            <div className="mb-8">
              <p className="text-[#555] text-xs tracking-[0.3em] uppercase mb-4">
                Predicted Kill Distribution
              </p>
              <div className="relative h-12 bg-[#111] w-full">
                
                {/* 10-90 range — lightest */}
                {(() => {
                  const min = result.killRange[0] - 2;
                  const max = result.killRange[1] + 2;
                  const range = max - min;
                  const leftPct = 0;
                  const widthPct = 100;
                  return (
                    <div className="absolute h-full bg-[#FF4655]/10"
                      style={{ left: `${leftPct}%`, width: `${widthPct}%` }} />
                  );
                })()}

                {/* 25-75 range — medium */}
                <div className="absolute h-full bg-[#FF4655]/25"
                  style={{
                    left: `25%`,
                    width: `50%`
                  }} />

                {/* Median marker */}
                <div className="absolute h-full w-0.5 bg-[#FF4655]"
                  style={{ left: `50%` }} />

                {/* Kill line marker */}
                {(() => {
                  const median = (result.killRange[0] + result.killRange[1]) / 2;
                  const spread = result.killRange[1] - result.killRange[0] + 4;
                  const min = median - spread;
                  const max = median + spread;
                  const pct = ((parseFloat(killLine) - min) / (max - min)) * 100;
                  const clampedPct = Math.min(Math.max(pct, 2), 98);
                  return (
                    <div className="absolute top-0 bottom-0 flex flex-col items-center"
                      style={{ left: `${clampedPct}%` }}>
                      <div className="text-[10px] text-white/60 -translate-x-1/2 mt-1">
                        {killLine}
                      </div>
                      <div className="w-2 h-2 rounded-full bg-white mt-1" />
                      <div className="w-px flex-1 bg-white/40" />
                    </div>
                  );
                })()}
              </div>

              {/* Labels */}
              <div className="flex justify-between mt-2 text-xs text-[#444]">
                <span>{result.killRange[0] - 2} kills</span>
                <span className="text-[#FF4655]">median: {Math.round((result.killRange[0] + result.killRange[1]) / 2)}</span>
                <span>{result.killRange[1] + 2} kills</span>
              </div>

              {/* Probabilities as text */}
              <div className="mt-4 space-y-1">
                <p className="text-xs text-[#888]">
                  Over {killLine}: <span className="text-[#FF4655] font-bold">{overPct}%</span>
                </p>
                <p className="text-xs text-[#888]">
                  Under {killLine}: <span className="text-white font-bold">{100 - overPct}%</span>
                </p>
              </div>
            </div>

            {/* Feature coefficients — QR only */}
            {model === "quantile_regression" && result.featureCoefficients && (
              <div className="mb-8 border-t border-[#1f1f1f] pt-8">
                <p className="text-[#FF4655] text-xs tracking-[0.3em] uppercase mb-1">
                  Top Feature Coefficients
                </p>
                <p className="text-[#2a2a2a] text-xs mb-4">
                  Median quantile (q=0.50) · higher = stronger predictor of kill count
                </p>
                <div className="space-y-3">
                  {result.featureCoefficients.map((f, i) => (
                    <div key={i}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[#888]">{f.feature}</span>
                        <span className="text-[#555]">{f.weight.toFixed(2)}</span>
                      </div>
                      <div className="h-1 bg-[#1a1a1a] w-full">
                        <div
                          className="h-full bg-[#FF4655] transition-all duration-500"
                          style={{
                            width: `${(f.weight / maxWeight) * 100}%`,
                            opacity: 1 - i * 0.1
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Similar players */}
            <div>
              <p className="text-[#555] text-xs tracking-[0.3em] uppercase mb-3">
                Similar Players via KNN
              </p>
              <div className="flex gap-3 flex-wrap">
                {result.similarPlayers.map((p, i) => (
                  <div key={i}
                    className="border border-[#1f1f1f] px-4 py-2 text-sm text-[#888]
                               hover:border-[#FF4655]/40 hover:text-white transition-colors cursor-pointer">
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
