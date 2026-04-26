import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const { player, map, killLine, model } = await req.json();

  const flaskRes = await fetch("http://127.0.0.1:5000/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ player, map, killLine, model })
});
const data = await flaskRes.json();
return NextResponse.json(data);

  const base = {
    killRange: [12, 18] as [number, number],
    overProbability: 0.63,
    archetype: "Entry Fragger",
    similarPlayers: ["aspas", "Derke", "nAts"],
  };

  // MLP — no feature coefficients (black box)
  if (model === "mlp") {
    await new Promise(r => setTimeout(r, 600));
    console.log(`[predict] model=mlp player=${player} map=${map} killLine=${killLine}`);
    return NextResponse.json({
      ...base,
      model: "MLP · 2-layer neural network · pinball loss",
    });
  }

  // Quantile Regression — return feature coefficients for interpretability
  await new Promise(r => setTimeout(r, 600));
  console.log(`[predict] model=qr player=${player} map=${map} killLine=${killLine}`);
  return NextResponse.json({
    ...base,
    killRange: [11, 17] as [number, number],
    overProbability: 0.58,
    model: "Quantile Regression · linear · pinball loss",
    featureCoefficients: [
      { feature: "ACS (Average Combat Score)", weight: 0.31 },
      { feature: "KAST %", weight: 0.24 },
      { feature: "ADR (Avg Damage / Round)", weight: 0.19 },
      { feature: "KPR (Kills / Round)", weight: 0.14 },
      { feature: "First Bloods", weight: 0.08 },
      { feature: "Agent Role", weight: 0.04 },
    ],
  });
}
