'use client';

import { useState, useEffect } from 'react';
import { QueryInput } from './components/QueryInput';
import { ResultsDisplay } from './components/ResultsDisplay';
import { ThemeProvider } from './components/ThemeProvider';
import { ThemeToggle } from './components/ThemeToggle';
import { motion } from 'framer-motion';

// Define TypeScript interfaces to match ResultsDisplay expectations
interface Response {
  model: string;
  responseText: string;
  timeMs: number;
  score: number;
}

interface ResultsData {
  responses: Response[];
  aggregateScore: number;
}

interface AggregateScores {
  [key: string]: number;
}

export default function Home() {
  const [results, setResults] = useState<ResultsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [aggregateScores, setAggregateScores] = useState<AggregateScores>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAggregateScores();
  }, []);

  const fetchAggregateScores = async (retries = 3, delay = 1000) => {
    for (let i = 0; i < retries; i++) {
      try {
        const response = await fetch('/llm/aggregateScores');
        if (!response.ok) throw new Error('Failed to fetch aggregate scores');
        const data = await response.json();
        setAggregateScores(data.aggregateScores);
        setError(null);
        return;
      } catch (error) {
        console.error('Error fetching aggregate scores:', error);
        if (i < retries - 1) await new Promise((resolve) => setTimeout(resolve, delay));
        else setError('Failed to load aggregate scores after multiple attempts.');
      }
    }
  };

  const handleQuerySubmit = async (query: string) => {
    setIsLoading(true);
    setResults(null);
    setError(null);
    try {
      const response = await fetch('/experiment/runOnePrompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ userPrompt: query }),
      });
      if (!response.ok) throw new Error('Failed to submit query');
      const data = await response.json();
      setResults(data);
      fetchAggregateScores();
    } catch (error) {
      console.error('Error submitting query:', error);
      setError('Failed to submit query. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <main className="container mx-auto p-4 min-h-screen flex flex-col">
        <nav className="flex justify-between items-center mb-8">
          <motion.h1
            className="text-3xl font-bold"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            LLM Evaluation Project
          </motion.h1>
          <ThemeToggle />
        </nav>

        {error && (
          <motion.div
            key="error-message"
            className="text-red-500 text-center mb-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {error}
          </motion.div>
        )}

        <QueryInput onSubmit={handleQuerySubmit} isLoading={isLoading} />

        {isLoading && (
          <motion.div
            key="loading-spinner"
            className="flex justify-center items-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full"
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 1, ease: 'linear', repeat: Infinity }}
            />
          </motion.div>
        )}

        {results && (
          <div key="results-display">
            <ResultsDisplay results={results} aggregateScores={aggregateScores} />
          </div>
        )}
      </main>
    </ThemeProvider>
  );
}