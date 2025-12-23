'use client';

import React, { ReactNode } from 'react';
import Link from 'next/link';
import { Spinner } from '../ui/Spinner';

// =============================================================================
// TYPES
// =============================================================================

export type UserTier = 'free' | 'pro';

export interface AuthGuardUser {
  id: string;
  email: string;
  tier: UserTier;
}

export interface AuthGuardProps {
  /** Content to protect */
  children: ReactNode;
  /** Content to show while loading */
  fallback?: ReactNode;
  /** Require authentication */
  requireAuth?: boolean;
  /** Required subscription tier */
  requiredTier?: UserTier;
  /** Current user (from your auth provider) */
  user?: AuthGuardUser | null;
  /** Loading state */
  isLoading?: boolean;
  /** Custom unauthorized component */
  unauthorizedComponent?: ReactNode;
  /** Custom upgrade component */
  upgradeComponent?: ReactNode;
}

// =============================================================================
// DEFAULT COMPONENTS
// =============================================================================

const DefaultFallback: React.FC = () => (
  <div className="min-h-[50vh] flex items-center justify-center">
    <Spinner size="lg" />
  </div>
);

const DefaultUnauthorized: React.FC = () => (
  <div className="min-h-[50vh] flex items-center justify-center p-8">
    <div className="text-center max-w-md">
      {/* Lock icon */}
      <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-cream-100 flex items-center justify-center">
        <svg
          className="w-8 h-8 text-navy-500"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
          />
        </svg>
      </div>

      <h2 className="font-heading text-heading-lg text-navy-900 mb-2">
        Sign in required
      </h2>
      <p className="text-body-md text-neutral-600 mb-6">
        Please sign in to access this feature. Create a free account to get started.
      </p>

      <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
        <Link
          href="/signin"
          className="w-full sm:w-auto px-6 py-3 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors"
        >
          Sign In
        </Link>
        <Link
          href="/signup"
          className="w-full sm:w-auto px-6 py-3 border border-border-medium text-navy-700 font-medium rounded-lg hover:bg-cream-50 transition-colors"
        >
          Create Account
        </Link>
      </div>
    </div>
  </div>
);

const DefaultUpgradeRequired: React.FC = () => (
  <div className="min-h-[50vh] flex items-center justify-center p-8">
    <div className="text-center max-w-md">
      {/* Pro badge icon */}
      <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-br from-terra-400 to-terra-600 flex items-center justify-center">
        <svg
          className="w-8 h-8 text-white"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 10V3L4 14h7v7l9-11h-7z"
          />
        </svg>
      </div>

      <h2 className="font-heading text-heading-lg text-navy-900 mb-2">
        Pro feature
      </h2>
      <p className="text-body-md text-neutral-600 mb-6">
        This feature is available on the Pro plan. Upgrade to unlock unlimited analyses, Quant Lab, and more.
      </p>

      <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
        <Link
          href="/pricing"
          className="w-full sm:w-auto px-6 py-3 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors"
        >
          View Plans
        </Link>
        <Link
          href="/"
          className="w-full sm:w-auto px-6 py-3 text-neutral-600 font-medium hover:text-navy-900 transition-colors"
        >
          Go Back
        </Link>
      </div>

      {/* Feature list */}
      <div className="mt-8 pt-8 border-t border-border-light">
        <p className="text-body-sm font-medium text-navy-900 mb-4">Pro includes:</p>
        <ul className="space-y-2 text-body-sm text-neutral-600">
          <li className="flex items-center justify-center gap-2">
            <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            Unlimited analyses
          </li>
          <li className="flex items-center justify-center gap-2">
            <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            Quant Lab access
          </li>
          <li className="flex items-center justify-center gap-2">
            <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            Export to CSV/PDF
          </li>
          <li className="flex items-center justify-center gap-2">
            <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            90-day history
          </li>
        </ul>
      </div>
    </div>
  </div>
);

// =============================================================================
// COMPONENT
// =============================================================================

export const AuthGuard: React.FC<AuthGuardProps> = ({
  children,
  fallback,
  requireAuth = true,
  requiredTier,
  user,
  isLoading = false,
  unauthorizedComponent,
  upgradeComponent,
}) => {
  // Show loading state
  if (isLoading) {
    return <>{fallback || <DefaultFallback />}</>;
  }

  // Check authentication
  if (requireAuth && !user) {
    return <>{unauthorizedComponent || <DefaultUnauthorized />}</>;
  }

  // Check tier requirement
  if (requiredTier === 'pro' && user?.tier !== 'pro') {
    return <>{upgradeComponent || <DefaultUpgradeRequired />}</>;
  }

  // Render protected content
  return <>{children}</>;
};

// =============================================================================
// FEATURE GATE (inline component for feature-level gating)
// =============================================================================

export interface FeatureGateProps {
  /** Content to show if allowed */
  children: ReactNode;
  /** Content to show if not allowed */
  fallback?: ReactNode;
  /** Required tier */
  requiredTier?: UserTier;
  /** Current user tier */
  userTier?: UserTier;
  /** Whether user is authenticated */
  isAuthenticated?: boolean;
}

export const FeatureGate: React.FC<FeatureGateProps> = ({
  children,
  fallback,
  requiredTier,
  userTier,
  isAuthenticated = false,
}) => {
  // Check if feature is accessible
  const hasAccess = () => {
    if (!isAuthenticated) return false;
    if (!requiredTier) return true;
    if (requiredTier === 'free') return true;
    if (requiredTier === 'pro') return userTier === 'pro';
    return false;
  };

  if (hasAccess()) {
    return <>{children}</>;
  }

  // Show fallback or nothing
  return fallback ? <>{fallback}</> : null;
};

export default AuthGuard;
