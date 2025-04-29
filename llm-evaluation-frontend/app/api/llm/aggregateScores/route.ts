import { NextResponse } from 'next/server';

export async function GET() {
  const aggregateScores = {
    ModelA: 4.5,
    ModelB: 3.9,
  };
  return NextResponse.json({ aggregateScores });
}