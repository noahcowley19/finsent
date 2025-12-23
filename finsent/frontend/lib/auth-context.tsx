'use client';

// =============================================================================
// AUTH CONTEXT PROVIDER
// =============================================================================
// Provides authentication state and methods to client components
//
// Usage:
//   // In root layout
//   <AuthProvider>{children}</AuthProvider>
//
//   // In components
//   const { user, signIn, signOut } = useAuth();
//
// =============================================================================

import React, {
  createContext,
  useContext,
  ReactNode,
  useCallback,
  useMemo,
} from 'react';
import { useSession, signIn as nextAuthSignIn, signOut as nextAuthSignOut } from 'next-auth/react';
import { SessionProvider } from 'next-auth/react';
import { useRouter } from 'next/navigation';

// =============================================================================
// TYPES
// =============================================================================

export interface AuthUser {
  id: string;
  email: string;
  name: string | null;
  image: string | null;
  tier: 'free' | 'pro';
}

export interface AuthContextValue {
  /** Current authenticated user */
  user: AuthUser | null;
  /** Whether auth state is loading */
  isLoading: boolean;
  /** Whether user is authenticated */
  isAuthenticated: boolean;
  /** Whether user has Pro subscription */
  isPro: boolean;
  /** Sign in with credentials */
  signIn: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  /** Sign in with OAuth provider */
  signInWithProvider: (provider: 'google' | 'github') => Promise<void>;
  /** Sign out */
  signOut: () => Promise<void>;
  /** Update session (refresh user data) */
  update: () => Promise<void>;
}

// =============================================================================
// CONTEXT
// =============================================================================

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

// =============================================================================
// HOOK
// =============================================================================

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// =============================================================================
// INNER PROVIDER (uses session)
// =============================================================================

function AuthContextProvider({ children }: { children: ReactNode }) {
  const { data: session, status, update: updateSession } = useSession();
  const router = useRouter();

  // Derive user from session
  const user = useMemo<AuthUser | null>(() => {
    if (!session?.user) return null;
    return {
      id: session.user.id,
      email: session.user.email!,
      name: session.user.name ?? null,
      image: session.user.image ?? null,
      tier: session.user.tier,
    };
  }, [session]);

  // Sign in with credentials
  const signIn = useCallback(async (email: string, password: string) => {
    try {
      const result = await nextAuthSignIn('credentials', {
        email,
        password,
        redirect: false,
      });

      if (result?.error) {
        return { success: false, error: result.error };
      }

      // Refresh the page to get new session
      router.refresh();
      return { success: true };
    } catch (error) {
      return { success: false, error: 'An unexpected error occurred' };
    }
  }, [router]);

  // Sign in with OAuth provider
  const signInWithProvider = useCallback(async (provider: 'google' | 'github') => {
    await nextAuthSignIn(provider, { callbackUrl: '/' });
  }, []);

  // Sign out
  const signOut = useCallback(async () => {
    await nextAuthSignOut({ callbackUrl: '/' });
  }, []);

  // Update session
  const update = useCallback(async () => {
    await updateSession();
  }, [updateSession]);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isLoading: status === 'loading',
      isAuthenticated: status === 'authenticated',
      isPro: user?.tier === 'pro',
      signIn,
      signInWithProvider,
      signOut,
      update,
    }),
    [user, status, signIn, signInWithProvider, signOut, update]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// =============================================================================
// MAIN PROVIDER (wraps SessionProvider)
// =============================================================================

export interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  return (
    <SessionProvider>
      <AuthContextProvider>{children}</AuthContextProvider>
    </SessionProvider>
  );
}

// =============================================================================
// UTILITY HOOKS
// =============================================================================

/**
 * Hook that requires authentication
 * Redirects to sign in if not authenticated
 */
export function useRequireAuth(redirectUrl = '/signin') {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  React.useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push(`${redirectUrl}?callbackUrl=${encodeURIComponent(window.location.pathname)}`);
    }
  }, [isAuthenticated, isLoading, router, redirectUrl]);

  return { isLoading, isAuthenticated };
}

/**
 * Hook that requires Pro subscription
 * Returns upgrade prompt state
 */
export function useRequirePro() {
  const { isPro, isAuthenticated, isLoading } = useAuth();
  const [showUpgrade, setShowUpgrade] = React.useState(false);

  const checkAccess = useCallback(() => {
    if (!isAuthenticated) {
      return { hasAccess: false, reason: 'unauthenticated' as const };
    }
    if (!isPro) {
      return { hasAccess: false, reason: 'requires_pro' as const };
    }
    return { hasAccess: true, reason: null };
  }, [isAuthenticated, isPro]);

  const requirePro = useCallback(() => {
    const { hasAccess } = checkAccess();
    if (!hasAccess) {
      setShowUpgrade(true);
      return false;
    }
    return true;
  }, [checkAccess]);

  return {
    isPro,
    isLoading,
    showUpgrade,
    setShowUpgrade,
    requirePro,
    checkAccess,
  };
}

export default AuthProvider;
