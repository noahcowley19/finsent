// =============================================================================
// PRISMA CLIENT SINGLETON
// =============================================================================
// Prevents multiple Prisma Client instances in development
// Handles missing DATABASE_URL gracefully during build
//
// Usage:
//   import { prisma } from '@/lib/prisma';
//   const users = await prisma.user.findMany();
//
// =============================================================================

import { PrismaClient } from '@prisma/client';

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

// Check if DATABASE_URL is available and valid (not a build placeholder)
const isDatabaseAvailable = 
  process.env.DATABASE_URL && 
  !process.env.DATABASE_URL.includes('build-placeholder-host');

// Only create Prisma Client if database is available
// During build with placeholder URL, this will be a dummy client
export const prisma =
  globalForPrisma.prisma ??
  (isDatabaseAvailable
    ? new PrismaClient({
        log: process.env.NODE_ENV === 'development' ? ['query', 'error', 'warn'] : ['error'],
      })
    : new PrismaClient({
        log: ['error'],
      }));

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma;
}

export default prisma;
