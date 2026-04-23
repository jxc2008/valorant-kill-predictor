import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const { player, map, killLine } = await req.json();

  // ── Dummy data for UI scaffolding ──────────────────────────────
  // Replace this block with a real call to app/api.py once Joseph's
  // Flask backend is running.
  const dummyResponse = {
    killRange: [12, 18] as [number, number],
    overProbability: 0.63,
    archetype: "Entry Fragger",
    similarPlayers: ["aspas", "Derke", "nAts"],
    model: "Quantile Regression · pinball loss",
  };

  // Simulate a small network delay so the loading state is visible
  await new Promise(r => setTimeout(r, 600));

  console.log(`[predict] player=${player} map=${map} killLine=${killLine}`);

  return NextResponse.json(dummyResponse);
}
