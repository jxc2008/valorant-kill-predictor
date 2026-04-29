import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON body" },
      { status: 400 }
    );
  }

  const { player, map, killLine, model } = body as {
    player?: string;
    map?: string;
    killLine?: number;
    model?: string;
  };

  let flaskRes: Response;
  try {
    flaskRes = await fetch("http://127.0.0.1:5000/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ player, map, killLine, model }),
    });
  } catch {
    return NextResponse.json(
      { error: "Prediction service unavailable. Is the Flask API running on :5000?" },
      { status: 503 }
    );
  }

  const text = await flaskRes.text();
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return NextResponse.json(
      { error: "Prediction service returned a non-JSON response" },
      { status: 502 }
    );
  }

  return NextResponse.json(data, { status: flaskRes.status });
}
