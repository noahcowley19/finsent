// =============================================================================
// NEXTAUTH API ROUTE
// =============================================================================
// Handles all NextAuth.js authentication endpoints
//
// Endpoints:
//   GET/POST /api/auth/signin
//   GET/POST /api/auth/signout
//   GET/POST /api/auth/callback/:provider
//   GET      /api/auth/session
//   GET      /api/auth/csrf
//   GET      /api/auth/providers
//
// =============================================================================

import NextAuth from 'next-auth';
import { authOptions } from '@/lib/auth';

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };
