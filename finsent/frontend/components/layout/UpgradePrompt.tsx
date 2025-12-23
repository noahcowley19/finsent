'use client';

import React, { ReactNode, HTMLAttributes } from 'react';
import Link from 'next/link';
import { Modal } from '../ui/Modal';

// =============================================================================
// TYPES
// =============================================================================

export interface UpgradePromptProps {
  /** Whether modal is open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Feature name being accessed */
  featureName?: string;
  /** Custom title */
  title?: string;
  /** Custom description */
  description?: string;
  /** Features included in Pro */
  features?: string[];
}

export interface UpgradeBannerProps extends HTMLAttributes<HTMLDivElement> {
  /** Feature name */
  featureName?: string;
  /** Dismissible */
  dismissible?: boolean;
  /** Dismiss handler */
  onDismiss?: () => void;
  /** Variant */
  variant?: 'default' | 'compact' | 'inline';
}

// =============================================================================
// DEFAULT FEATURES
// =============================================================================

const defaultProFeatures = [
  'Unlimited stock analyses',
  'Quant Lab access',
  'Unlimited portfolio positions',
  'Unlimited watchlist stocks',
  'Export to CSV/PDF',
  'Email alerts',
  '90-day analysis history',
  'Priority support',
];

// =============================================================================
// UPGRADE PROMPT MODAL
// =============================================================================

export const UpgradePrompt: React.FC<UpgradePromptProps> = ({
  isOpen,
  onClose,
  featureName = 'this feature',
  title,
  description,
  features = defaultProFeatures,
}) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      size="md"
    >
      <div className="text-center">
        {/* Icon */}
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

        {/* Title */}
        <h2 className="font-heading text-heading-lg text-navy-900 mb-2">
          {title || `Unlock ${featureName}`}
        </h2>

        {/* Description */}
        <p className="text-body-md text-neutral-600 mb-6">
          {description || `Upgrade to Pro to access ${featureName} and unlock the full power of Caveray.`}
        </p>

        {/* Price */}
        <div className="mb-6 p-4 bg-cream-50 rounded-lg">
          <div className="flex items-baseline justify-center gap-1">
            <span className="text-display-sm font-heading text-navy-900">$9.99</span>
            <span className="text-body-md text-neutral-500">/month</span>
          </div>
          <p className="text-body-sm text-neutral-500 mt-1">Cancel anytime</p>
        </div>

        {/* Features */}
        <div className="text-left mb-6">
          <p className="text-body-sm font-medium text-navy-900 mb-3">Everything in Pro:</p>
          <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {features.slice(0, 6).map((feature, index) => (
              <li key={index} className="flex items-center gap-2 text-body-sm text-neutral-600">
                <svg className="w-4 h-4 text-success-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                {feature}
              </li>
            ))}
          </ul>
        </div>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-3">
          <Link
            href="/pricing"
            className="flex-1 px-6 py-3 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors text-center"
          >
            Upgrade to Pro
          </Link>
          <button
            onClick={onClose}
            className="flex-1 px-6 py-3 text-neutral-600 font-medium hover:text-navy-900 transition-colors"
          >
            Maybe Later
          </button>
        </div>
      </div>
    </Modal>
  );
};

// =============================================================================
// UPGRADE BANNER
// =============================================================================

export const UpgradeBanner: React.FC<UpgradeBannerProps> = ({
  featureName,
  dismissible = true,
  onDismiss,
  variant = 'default',
  className = '',
  ...props
}) => {
  if (variant === 'compact') {
    return (
      <div
        className={`
          flex items-center justify-between gap-4 p-3
          bg-gradient-to-r from-terra-50 to-terra-100
          border border-terra-200 rounded-lg
          ${className}
        `}
        {...props}
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-terra-500 flex items-center justify-center flex-shrink-0">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <p className="text-body-sm text-terra-800">
            <span className="font-medium">Upgrade to Pro</span>
            {featureName && ` to unlock ${featureName}`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/pricing"
            className="px-3 py-1.5 text-body-sm font-medium text-white bg-terra-500 rounded hover:bg-terra-600 transition-colors"
          >
            Upgrade
          </Link>
          {dismissible && onDismiss && (
            <button
              onClick={onDismiss}
              className="p-1 text-terra-400 hover:text-terra-600 transition-colors"
              aria-label="Dismiss"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
    );
  }

  if (variant === 'inline') {
    return (
      <span className={`inline-flex items-center gap-1.5 ${className}`} {...props}>
        <svg className="w-4 h-4 text-terra-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
        <Link href="/pricing" className="text-body-sm font-medium text-terra-500 hover:text-terra-600 transition-colors">
          Pro feature
        </Link>
      </span>
    );
  }

  // Default variant
  return (
    <div
      className={`
        relative overflow-hidden
        bg-gradient-to-r from-terra-500 to-terra-600
        rounded-xl p-6 text-white
        ${className}
      `}
      {...props}
    >
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-10">
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          <defs>
            <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
              <path d="M 10 0 L 0 0 0 10" fill="none" stroke="white" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100" height="100" fill="url(#grid)" />
        </svg>
      </div>

      <div className="relative flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-white/20 flex items-center justify-center flex-shrink-0">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div>
            <h3 className="font-heading font-semibold text-lg">Upgrade to Pro</h3>
            <p className="text-white/80 text-body-sm mt-1">
              {featureName 
                ? `Unlock ${featureName} and get unlimited access to all features.`
                : 'Get unlimited access to all features and take your analysis to the next level.'
              }
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3 flex-shrink-0">
          <Link
            href="/pricing"
            className="px-5 py-2.5 bg-white text-terra-600 font-medium rounded-lg hover:bg-cream-50 transition-colors"
          >
            View Plans
          </Link>
          {dismissible && onDismiss && (
            <button
              onClick={onDismiss}
              className="p-2 text-white/60 hover:text-white transition-colors"
              aria-label="Dismiss"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// LOCKED FEATURE OVERLAY
// =============================================================================

export interface LockedFeatureOverlayProps extends HTMLAttributes<HTMLDivElement> {
  /** Feature name */
  featureName?: string;
  /** Show upgrade button */
  showUpgradeButton?: boolean;
}

export const LockedFeatureOverlay: React.FC<LockedFeatureOverlayProps> = ({
  featureName = 'this feature',
  showUpgradeButton = true,
  children,
  className = '',
  ...props
}) => {
  return (
    <div className={`relative ${className}`} {...props}>
      {/* Blurred content */}
      <div className="blur-sm pointer-events-none select-none" aria-hidden="true">
        {children}
      </div>

      {/* Overlay */}
      <div className="absolute inset-0 bg-cream-50/80 backdrop-blur-[2px] flex items-center justify-center">
        <div className="text-center p-6">
          <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-cream-100 flex items-center justify-center">
            <svg className="w-6 h-6 text-navy-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <p className="text-body-sm font-medium text-navy-900 mb-1">Pro Feature</p>
          <p className="text-caption text-neutral-500 mb-4">
            Upgrade to unlock {featureName}
          </p>
          {showUpgradeButton && (
            <Link
              href="/pricing"
              className="inline-flex px-4 py-2 text-body-sm font-medium text-white bg-terra-500 rounded-lg hover:bg-terra-600 transition-colors"
            >
              Upgrade
            </Link>
          )}
        </div>
      </div>
    </div>
  );
};

export default UpgradePrompt;
