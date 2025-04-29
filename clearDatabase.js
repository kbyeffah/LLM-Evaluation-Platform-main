// clearDatabase.js
const { PrismaClient } = require("@prisma/client");

const prisma = new PrismaClient();

async function main() {
  console.log("Clearing database...");

  // The order of deletion is important to respect foreign key constraints
  await prisma.result.deleteMany({});
  await prisma.experimentRun.deleteMany({});
  await prisma.experimentTestCase.deleteMany({});
  await prisma.testCase.deleteMany({});
  await prisma.experiment.deleteMany({});

  console.log("Database cleared successfully!");
}

main()
  .catch((e) => {
    console.error("Error clearing the database:", e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
