// =============================================================================
// NEXTAUTH.JS CONFIGURATION
// =============================================================================
// Authentication configuration with credentials and optional OAuth providers
// Handles build-time environment gracefully when DATABASE_URL is not available
//
// Environment variables required (runtime):
//   - NEXTAUTH_URL: Your site URL
//   - NEXTAUTH_SECRET: Random secret for JWT encryption
//   - DATABASE_URL: PostgreSQL connection string
//
// =============================================================================

import { NextAuthOptions } from 'next-auth';
import { PrismaAdapter } from '@auth/prisma-adapter';
import CredentialsProvider from 'next-auth/providers/credentials';
import GoogleProvider from 'next-auth/providers/google';
import bcrypt from 'bcryptjs';
import { prisma } from './prisma';

// =============================================================================
// TYPE EXTENSIONS
// =============================================================================

declare module 'next-auth' {
  interface Session {
    user: {
      id: string;
      email: string;
      name: string | null;
      image: string | null;
      tier: 'free' | 'pro';
    };
  }

  interface User {
    id: string;
    email: string;
    name: string | null;
    image: string | null;
    tier: 'free' | 'pro';
  }
}

declare module 'next-auth/jwt' {
  interface JWT {
    id: string;
    tier: 'free' | 'pro';
  }
}

// =============================================================================
// HELPER: Check if database is available (not a build placeholder)
// =============================================================================

const isDatabaseAvailable = 
  process.env.DATABASE_URL && 
  !process.env.DATABASE_URL.includes('build-placeholder-host');

// =============================================================================
// AUTH OPTIONS
// =============================================================================

export const authOptions: NextAuthOptions = {
  // Only use Prisma adapter if database is actually available
  // During build with placeholder, adapter will be undefined (JWT-only mode)
  adapter: isDatabaseAvailable 
    ? (PrismaAdapter(prisma) as NextAuthOptions['adapter'])
    : undefined,
  
  session: {
    strategy: 'jwt',
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },

  pages: {
    signIn: '/signin',
    signOut: '/signout',
    error: '/signin', // Error code passed in query string as ?error=
    newUser: '/welcome', // New users will be directed here on first sign in
  },

  providers: [
    // Credentials Provider (email/password)
    CredentialsProvider({
      id: 'credentials',
      name: 'Email',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        // During build time, don't attempt database operations
        if (!isDatabaseAvailable) {
          console.warn('Database not available - skipping auth during build');
          return null;
        }

        if (!credentials?.email || !credentials?.password) {
          throw new Error('Email and password are required');
        }

        const user = await prisma.user.findUnique({
          where: { email: credentials.email.toLowerCase() },
        });

        if (!user || !user.password) {
          throw new Error('Invalid email or password');
        }

        const isValidPassword = await bcrypt.compare(
          credentials.password,
          user.password
        );

        if (!isValidPassword) {
          throw new Error('Invalid email or password');
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          image: user.image,
          tier: user.tier.toLowerCase() as 'free' | 'pro',
        };
      },
    }),

    // Google Provider (optional - uncomment and add env vars to enable)
    // GoogleProvider({
    //   clientId: process.env.GOOGLE_CLIENT_ID!,
    //   clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    // }),
  ],

  callbacks: {
    async jwt({ token, user, trigger, session }) {
      // Initial sign in
      if (user) {
        token.id = user.id;
        token.tier = user.tier;
      }

      // Handle session update (e.g., after subscription change)
      if (trigger === 'update' && session) {
        token.tier = session.tier;
      }

      // Refresh user data from database periodically (only if DB is available)
      if (token.id && isDatabaseAvailable) {
        try {
          const dbUser = await prisma.user.findUnique({
            where: { id: token.id },
            select: { tier: true, name: true, image: true },
          });

          if (dbUser) {
            token.tier = dbUser.tier.toLowerCase() as 'free' | 'pro';
            token.name = dbUser.name;
            token.picture = dbUser.image;
          }
        } catch (error) {
          // If database is unavailable at runtime, log but don't fail
          console.error('Failed to refresh user data from database:', error);
        }
      }

      return token;
    },

    async session({ session, token }) {
      if (token && session.user) {
        session.user.id = token.id;
        session.user.tier = token.tier;
      }
      return session;
    },

    async signIn({ user, account }) {
      // Allow OAuth without email verification
      if (account?.provider !== 'credentials') {
        return true;
      }

      // For credentials, just allow sign in
      // Add email verification check here if needed
      return true;
    },
  },

  events: {
    async signIn({ user, isNewUser }) {
      if (isNewUser) {
        // Track new user sign up
        console.log(`New user signed up: ${user.email}`);
      }
    },
  },

  debug: process.env.NODE_ENV === 'development',
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Hash a password using bcrypt
 */
export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, 12);
}

/**
 * Compare a password with a hash
 */
export async function verifyPassword(
  password: string,
  hashedPassword: string
): Promise<boolean> {
  return bcrypt.compare(password, hashedPassword);
}

/**
 * Generate a random token for password reset
 */
export function generateToken(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let token = '';
  for (let i = 0; i < 32; i++) {
    token += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return token;
}

export default authOptions;
