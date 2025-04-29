import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  const { userPrompt } = await request.json();
  const responses = [
    {
      model: 'ModelA',
      responseText: `Processed: ${userPrompt}`,
      timeMs: 150,
      score: 4.2,
    },
    {
      model: 'ModelB',
      responseText: `Alternative: ${userPrompt}`,
      timeMs: 200,
      score: 3.8,
    },
  ];
  const aggregateScore = responses.reduce((sum, r) => sum + r.score, 0) / responses.length;
  return NextResponse.json({ responses, aggregateScore });
}