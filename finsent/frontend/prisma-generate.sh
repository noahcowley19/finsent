#!/bin/bash
# Graceful Prisma generation script that doesn't fail if DATABASE_URL is missing
# This is needed for Netlify builds where we don't have a database connection

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
  echo "⚠️  DATABASE_URL not set. Using placeholder for build-time Prisma generation only."
  echo "⚠️  THIS IS NOT A REAL DATABASE - Only used to generate Prisma Client during build."
  # Use an obviously fake placeholder that won't be mistaken for production
  export DATABASE_URL="postgresql://build-user:build-pass@build-placeholder-host:5432/build-placeholder-db"
fi

# Generate Prisma Client
echo "Generating Prisma Client..."
npx prisma generate

echo "✅ Prisma Client generated successfully"
