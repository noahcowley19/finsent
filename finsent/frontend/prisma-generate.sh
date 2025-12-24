#!/bin/bash
# Graceful Prisma generation script that doesn't fail if DATABASE_URL is missing
# This is needed for Netlify builds where we don't have a database connection

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
  echo "⚠️  DATABASE_URL not set. Using placeholder for build."
  export DATABASE_URL="postgresql://placeholder:placeholder@localhost:5432/placeholder"
fi

# Generate Prisma Client
echo "Generating Prisma Client..."
npx prisma generate

echo "✅ Prisma Client generated successfully"
