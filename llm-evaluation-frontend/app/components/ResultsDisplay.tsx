'use client';

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { motion, useAnimation } from "framer-motion";
import { Progress } from "@/components/ui/progress";
import ReactMarkdown from 'react-markdown';

interface Response {
  model: string;
  responseText: string;
  timeMs: number;
  score: number;
}

interface ResultsDisplayProps {
  results: {
    responses: Response[];
    aggregateScore: number;
  };
  aggregateScores: {
    [key: string]: number;
  };
}

export function ResultsDisplay({
  results,
  aggregateScores,
}: ResultsDisplayProps) {
  // Explicitly type animatedScores as number[]
  const [animatedScores, setAnimatedScores] = useState<number[]>(
    results?.responses?.map(() => 0) ?? []
  );

  // Explicitly type animatedAggregateScores as { [key: string]: number }
  const [animatedAggregateScores, setAnimatedAggregateScores] = useState<{
    [key: string]: number;
  }>(() => {
    // Initialize with all response.model keys to ensure they exist
    const initialScores: { [key: string]: number } = {};
    // Add keys from aggregateScores
    Object.keys(aggregateScores ?? {}).forEach((key) => {
      initialScores[key] = 0;
    });
    // Add keys from results.responses to handle new models
    results?.responses?.forEach((response) => {
      if (!(response.model in initialScores)) {
        initialScores[response.model] = 0;
      }
    });
    return initialScores;
  });

  const controls = useAnimation();

  useEffect(() => {
    const animateScores = async () => {
      await controls.start({ opacity: 1, y: 0 });
      for (let i = 0; i <= 100; i++) {
        setAnimatedScores(
          (results?.responses ?? []).map(
            (response) => (response.score / 5) * (i / 100) * 100
          )
        );
        setAnimatedAggregateScores(
          Object.entries(aggregateScores ?? {}).reduce(
            (acc, [key, value]) => ({
              ...acc,
              [key]: (value / 5) * (i / 100) * 100,
            }),
            // Ensure all response.model keys are included
            results?.responses?.reduce(
              (acc: { [key: string]: number }, response) => ({
                ...acc,
                [response.model]: acc[response.model] ?? 0,
              }),
              {}
            ) ?? {}
          )
        );
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
    };
    animateScores();
  }, [results, aggregateScores, controls]);

  if (!results?.responses?.length) {
    return <div>No results available.</div>;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={controls}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-2xl font-semibold mb-4">Results</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {results.responses.map((response, index) => (
          <motion.div
            key={response.model || index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <Card className="h-full flex flex-col overflow-hidden group">
              <CardHeader>
                <CardTitle className="flex justify-between items-center">
                  <Badge variant={getScoreVariant(response.score)}>
                    {response.model}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-grow flex flex-col">
                <div className="mb-4 flex-grow overflow-y-auto prose prose-sm dark:prose-invert">
                  <ReactMarkdown>{response.responseText}</ReactMarkdown>
                </div>
                <div className="mt-auto">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-medium">
                      Score: {response.score.toFixed(2)}/5
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {(animatedScores[index] / 20).toFixed(2)}/5
                    </span>
                  </div>
                  <Progress
                    value={animatedScores[index] ?? 0}
                    className="h-2 mb-2"
                  />
                  <p className="text-sm text-muted-foreground">
                    Response time: {response.timeMs}ms
                  </p>
                </div>
                <div className="mt-4">
                  <h3 className="text-sm font-semibold mb-2">
                    Aggregate Score
                  </h3>
                  <CircularProgress
                    progress={(aggregateScores?.[response.model] ?? 0) / 5}
                    animatedProgress={
                      (animatedAggregateScores[response.model] ?? 0) / 100
                    }
                  />
                </div>
              </CardContent>
              <div className="h-1 bg-gradient-to-r from-primary to-secondary transform origin-left scale-x-0 transition-transform duration-300 ease-in-out group-hover:scale-x-100" />
            </Card>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function getScoreVariant(score: number) {
  if (score >= 4) return "default";
  if (score >= 3) return "secondary";
  return "destructive";
}

function CircularProgress({
  progress,
  animatedProgress,
}: {
  progress: number;
  animatedProgress: number;
}) {
  const circumference = 2 * Math.PI * 60;

  return (
    <div className="relative w-24 h-24">
      <svg className="w-full h-full" viewBox="0 0 128 128">
        <circle
          className="text-muted-foreground"
          strokeWidth="8"
          stroke="currentColor"
          fill="transparent"
          r="60"
          cx="64"
          cy="64"
        />
        <circle
          className="text-primary"
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={circumference - animatedProgress * circumference}
          strokeLinecap="round"
          stroke="currentColor"
          fill="transparent"
          r="60"
          cx="64"
          cy="64"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-lg font-bold">{(progress * 5).toFixed(2)}</span>
      </div>
    </div>
  );
}