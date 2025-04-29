import express from "express";
import cors from "cors";
import { PrismaClient } from "@prisma/client";
import { Groq } from "groq-sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";
import axios from "axios";
import dotenv from "dotenv";

dotenv.config();
console.log("GROQ_API_KEY:", process.env.GROQ_API_KEY);
console.log("GEMINI_API_KEY:", process.env.GEMINI_API_KEY);
console.log("TOGETHER_API_KEY:", process.env.TOGETHER_API_KEY);
console.log("DATABASE_URL:", process.env.DATABASE_URL);

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const prisma = new PrismaClient();

const app = express();
app.use(cors({ origin: "http://localhost:3000" }));
app.use(express.json());

// Evaluation function to score responses using Gemini
async function evaluateResponseWithGemini(prompt, response, model) {
  const evaluationPrompt = `
You are an expert AI model evaluator. Rate the following AI response on a scale of 1-5 (where 5 is best), 
considering accuracy, relevance, and clarity.

Original prompt: "${prompt}"
Response from ${model}: "${response}"

Provide only a single number (1-5) as your response.`;

  try {
    const evalStartTime = Date.now();
    const result = await genAI
      .getGenerativeModel({ model: "gemini-1.5-flash" })
      .generateContent(evaluationPrompt);
    const evalTimeMs = Date.now() - evalStartTime;
    console.log(`Gemini evaluation for ${model} took ${evalTimeMs}ms`);
    const score = parseInt(result.response.text().trim());
    return isNaN(score) ? 3 : Math.min(Math.max(score, 1), 5);
  } catch (error) {
    console.error("Gemini evaluation error:", error.message, error.stack);
    return 3;
  }
}

app.get("/health", (req, res) => {
  res.json({ status: "OK" });
});

app.post("/experiment/runOnePrompt", async (req, res) => {
  console.log("Received /experiment/runOnePrompt request:", req.body);
  const { userPrompt } = req.body;
  if (!userPrompt) {
    console.error("Missing userPrompt in request body");
    return res.status(400).json({ error: "Missing 'userPrompt' in request body" });
  }

  try {
    console.log("Checking API keys...");
    if (!process.env.GROQ_API_KEY || !process.env.GEMINI_API_KEY || !process.env.TOGETHER_API_KEY) {
      console.warn("One or more API keys missing. Generating and saving mock data.");

      const experiment = await prisma.experiment.create({
        data: {
          name: "Mock Experiment",
          systemPrompt: "Mock system prompt",
          modelName: "mock-model",
        },
      });

      const testCase = await prisma.testCase.create({
        data: {
          userMessage: userPrompt,
          expectedOutput: "Mock expected output",
          graderType: "mock-grader",
        },
      });

      const experimentRun = await prisma.experimentRun.create({
        data: {
          experimentId: experiment.id,
          runName: "Mock Run",
          startedAt: new Date(),
          completedAt: new Date(),
          aggregateScore: null,
        },
      });

      const responses = [
        { model: "mock-llama3-70b-8192", responseText: `Mock: ${userPrompt}`, timeMs: 150, score: 4.0 },
        { model: "mock-gemini-1.5-flash", responseText: `Mock: ${userPrompt}`, timeMs: 200, score: 3.5 },
        { model: "mock-mixtral-8x7b-instruct-v0.1", responseText: `Mock: ${userPrompt}`, timeMs: 180, score: 3.8 },
      ];

      for (const response of responses) {
        await prisma.result.create({
          data: {
            experimentRunId: experimentRun.id,
            testCaseId: testCase.id,
            llmResponse: response.responseText,
            score: response.score,
            graderDetails: `Mock grader details for ${response.model}`,
          },
        });
      }

      const avgScore = responses.reduce((sum, r) => sum + r.score, 0) / responses.length;

      await prisma.experimentRun.update({
        where: { id: experimentRun.id },
        data: { aggregateScore: avgScore },
      });

      return res.json({
        experimentId: experiment.id,
        experimentRunId: experimentRun.id,
        testCaseId: testCase.id,
        responses,
        aggregateScore: avgScore,
      });
    }

    console.log("API keys found. Running real experiment...");

    const experiment = await prisma.experiment.create({
      data: {
        name: "Real Experiment",
        systemPrompt: "You are a helpful AI assistant.",
        modelName: "multi-model",
      },
    });

    const testCase = await prisma.testCase.create({
      data: {
        userMessage: userPrompt,
        expectedOutput: "Expected output TBD",
        graderType: "auto",
      },
    });

    const experimentRun = await prisma.experimentRun.create({
      data: {
        experimentId: experiment.id,
        runName: "Real Run",
        startedAt: new Date(),
      },
    });

    // Run API calls in parallel
    const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
    const geminiClient = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    const [groqResponse, geminiResponse, togetherResponse] = await Promise.all([
      (async () => {
        const startTime = Date.now();
        const response = await groq.chat.completions.create({
          model: "llama3-70b-8192",
          messages: [
            { role: "system", content: experiment.systemPrompt },
            { role: "user", content: userPrompt },
          ],
          max_tokens: 500,
        });
        const timeMs = Date.now() - startTime;
        console.log(`Groq API call took ${timeMs}ms`);
        return { response, timeMs };
      })(),
      (async () => {
        const startTime = Date.now();
        const geminiPrompt = `${experiment.systemPrompt} Provide a detailed explanation of the following: ${userPrompt}. Include classifications, examples, and additional context where applicable.`;
        const response = await geminiClient.generateContent(geminiPrompt, {
          maxOutputTokens: 500,
          temperature: 0.9,
        });
        const timeMs = Date.now() - startTime;
        console.log(`Gemini API call took ${timeMs}ms`);
        return { response, timeMs };
      })(),
      (async () => {
        const startTime = Date.now();
        const response = await axios.post(
          "https://api.together.xyz/inference",
          {
            model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
            prompt: `${experiment.systemPrompt} ${userPrompt}`,
            temperature: 0.7,
            top_p: 0.7,
            max_tokens: 512,
          },
          {
            headers: {
              Authorization: `Bearer ${process.env.TOGETHER_API_KEY}`,
              "Content-Type": "application/json",
            },
            timeout: 30000,
          }
        );
        const timeMs = Date.now() - startTime;
        console.log(`Together AI API call took ${timeMs}ms`);
        console.log("Together AI raw response:", JSON.stringify(response.data, null, 2));
        if (
          !response.data ||
          !response.data.choices ||
          !response.data.choices[0] ||
          typeof response.data.choices[0].text !== "string"
        ) {
          throw new Error("Unexpected Together AI response format: 'choices[0].text' is missing or not a string");
        }
        return { response, timeMs };
      })(),
    ]);

    const groqResult = {
      model: "llama3-70b-8192",
      responseText: groqResponse.response.choices[0].message.content,
      timeMs: groqResponse.timeMs,
      score: null,
    };

    const geminiResult = {
      model: "gemini-1.5-flash",
      responseText: geminiResponse.response.response.text(),
      timeMs: geminiResponse.timeMs,
      score: null,
    };

    const togetherResult = {
      model: "mixtral-8x7b-instruct-v0.1",
      responseText: togetherResponse.response.data.choices[0].text,
      timeMs: togetherResponse.timeMs,
      score: null,
    };

    // Evaluate responses using Gemini (also in parallel)
    const [groqScore, geminiScore, togetherScore] = await Promise.all([
      evaluateResponseWithGemini(userPrompt, groqResult.responseText, groqResult.model),
      evaluateResponseWithGemini(userPrompt, geminiResult.responseText, geminiResult.model),
      evaluateResponseWithGemini(userPrompt, togetherResult.responseText, togetherResult.model),
    ]);

    groqResult.score = groqScore;
    console.log(`Groq score for ${groqResult.model}: ${groqResult.score}`);
    geminiResult.score = geminiScore;
    console.log(`Gemini score for ${geminiResult.model}: ${geminiResult.score}`);
    togetherResult.score = togetherScore;
    console.log(`Together AI score for ${togetherResult.model}: ${togetherResult.score}`);

    const responses = [groqResult, geminiResult, togetherResult];

    // Save results to database
    for (const response of responses) {
      await prisma.result.create({
        data: {
          experimentRunId: experimentRun.id,
          testCaseId: testCase.id,
          llmResponse: response.responseText,
          score: response.score,
          graderDetails: `Graded by Gemini for ${response.model}`,
        },
      });
    }

    const avgScore = responses.reduce((sum, r) => sum + r.score, 0) / responses.length;
    console.log(`Calculated aggregate score: ${avgScore}`);

    await prisma.experimentRun.update({
      where: { id: experimentRun.id },
      data: {
        completedAt: new Date(),
        aggregateScore: avgScore,
      },
    });

    return res.json({
      experimentId: experiment.id,
      experimentRunId: experimentRun.id,
      testCaseId: testCase.id,
      responses,
      aggregateScore: avgScore,
    });
  } catch (error) {
    console.error("Error in /experiment/runOnePrompt:", error.message, error.stack);
    res.status(500).json({ error: `Failed to run experiment prompt: ${error.message}` });
  }
});

app.get("/llm/aggregateScores", async (req, res) => {
  console.log("Received /llm/aggregateScores request:", {
    headers: req.headers,
    query: req.query,
    method: req.method,
  });

  try {
    console.log("Starting /llm/aggregateScores endpoint...");
    console.log("Testing Prisma connection...");
    await prisma.$connect();
    console.log("Prisma connection successful");

    console.log("Fetching results from the database...");
    const results = await prisma.result.findMany();

    console.log("Results fetched:", results);

    if (!results || results.length === 0) {
      console.log("No results found. Returning empty aggregate scores.");
      return res.json({ aggregateScores: {} });
    }

    const aggregateScores = {};
    for (const result of results) {
      let modelName = "unknown";
      if (result.graderDetails) {
        const graderDetailsLower = result.graderDetails.toLowerCase();
        if (graderDetailsLower.includes("llama3-70b-8192")) {
          modelName = "llama3-70b-8192";
        } else if (graderDetailsLower.includes("gemini-1.5-flash")) {
          modelName = "gemini-1.5-flash";
        } else if (graderDetailsLower.includes("mixtral-8x7b-instruct-v0.1")) {
          modelName = "mixtral-8x7b-instruct-v0.1";
        }
      }

      console.log(`Result ${result.id}: modelName=${modelName}, score=${result.score}`);

      if (modelName === "unknown") {
        console.warn(`Skipping result ${result.id}: Unknown model (graderDetails: ${result.graderDetails})`);
        continue;
      }

      if (result.score != null && typeof result.score === "number") {
        if (!aggregateScores[modelName]) {
          aggregateScores[modelName] = { totalScore: 0, count: 0 };
        }
        aggregateScores[modelName].totalScore += result.score;
        aggregateScores[modelName].count += 1;
      } else {
        console.warn(`Skipping result ${result.id}: Invalid or missing score`);
      }
    }

    console.log("Raw aggregate scores:", aggregateScores);

    for (const modelName in aggregateScores) {
      const { totalScore, count } = aggregateScores[modelName];
      aggregateScores[modelName] = count > 0 ? totalScore / count : 0;
    }

    console.log("Final aggregate scores:", aggregateScores);

    res.json({ aggregateScores });
  } catch (error) {
    console.error("Error in /llm/aggregateScores:", error.message, error.stack);
    res.status(500).json({ error: "Failed to calculate aggregate scores for LLMs" });
  } finally {
    await prisma.$disconnect();
    console.log("Prisma connection disconnected");
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});