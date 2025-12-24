#!/bin/bash
# Graceful Prisma generation script that doesn't fail if DATABASE_URL is missing
# This is needed for Netlify builds where we don't have a database connection

set -e  # Exit on error

# Check if we should skip Prisma generation entirely
if [ "$SKIP_PRISMA_GENERATE" = "true" ] || [ "$PRISMA_GENERATE_SKIP" = "true" ]; then
  echo "⏭️  Skipping Prisma generation (SKIP_PRISMA_GENERATE or PRISMA_GENERATE_SKIP is set)"
  exit 0
fi

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
  echo "⚠️  DATABASE_URL not set. Using placeholder for build-time Prisma generation only."
  echo "⚠️  THIS IS NOT A REAL DATABASE - Only used to generate Prisma Client during build."
  echo "⚠️  Runtime database operations will fail without a real DATABASE_URL."
  
  # Use an obviously fake placeholder that won't be mistaken for production
  export DATABASE_URL="postgresql://build-user:build-pass@build-placeholder-host:5432/build-placeholder-db"
fi

# Generate Prisma Client
echo "🔧 Generating Prisma Client..."
if npx prisma generate; then
  echo "✅ Prisma Client generated successfully"
else
  # If generation fails, provide helpful error message
  echo "❌ Prisma Client generation failed"
  echo "   This might be due to:"
  echo "   - Invalid Prisma schema syntax"
  echo "   - Missing Prisma CLI"
  echo "   - Insufficient permissions"
  exit 1
fi
