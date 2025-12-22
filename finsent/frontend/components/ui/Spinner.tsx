'use client';

import React, { HTMLAttributes } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type SpinnerSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
export type SpinnerVariant = 'default' | 'primary' | 'white';

export interface SpinnerProps extends HTMLAttributes<HTMLDivElement> {
  /** Spinner size */
  size?: SpinnerSize;
  /** Color variant */
  variant?: SpinnerVariant;
  /** Accessible label */
  label?: string;
}

// =============================================================================
// STYLES
// =============================================================================

const sizeStyles: Record<SpinnerSize, { container: string; border: string }> = {
  xs: { container: 'w-3 h-3', border: 'border' },
  sm: { container: 'w-4 h-4', border: 'border-2' },
  md: { container: 'w-6 h-6', border: 'border-2' },
  lg: { container: 'w-8 h-8', border: 'border-[3px]' },
  xl: { container: 'w-12 h-12', border: 'border-4' },
};

const variantStyles: Record<SpinnerVariant, { track: string; indicator: string }> = {
  default: { track: 'border-cream-200', indicator: 'border-t-navy-500' },
  primary: { track: 'border-terra-200', indicator: 'border-t-terra-500' },
  white: { track: 'border-white/30', indicator: 'border-t-white' },
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Spinner: React.FC<SpinnerProps> = ({
  size = 'md',
  variant = 'default',
  label = 'Loading',
  className = '',
  ...props
}) => {
  const { container, border } = sizeStyles[size];
  const { track, indicator } = variantStyles[variant];

  return (
    <div
      role="status"
      aria-label={label}
      className={`inline-flex items-center justify-center ${className}`}
      {...props}
    >
      <div
        className={`
          ${container}
          ${border}
          ${track}
          ${indicator}
          rounded-full
          animate-spin
        `}
      />
      <span className="sr-only">{label}</span>
    </div>
  );
};

// =============================================================================
// LOADING DOTS
// =============================================================================

export interface LoadingDotsProps extends HTMLAttributes<HTMLDivElement> {
  /** Dot size */
  size?: 'sm' | 'md' | 'lg';
  /** Color variant */
  variant?: SpinnerVariant;
}

export const LoadingDots: React.FC<LoadingDotsProps> = ({
  size = 'md',
  variant = 'default',
  className = '',
  ...props
}) => {
  const dotSizes = {
    sm: 'w-1.5 h-1.5',
    md: 'w-2 h-2',
    lg: 'w-3 h-3',
  };

  const dotColors = {
    default: 'bg-navy-500',
    primary: 'bg-terra-500',
    white: 'bg-white',
  };

  return (
    <div
      role="status"
      className={`inline-flex items-center gap-1 ${className}`}
      {...props}
    >
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={`
            ${dotSizes[size]}
            ${dotColors[variant]}
            rounded-full
            animate-pulse
          `}
          style={{
            animationDelay: `${i * 150}ms`,
            animationDuration: '1s',
          }}
        />
      ))}
      <span className="sr-only">Loading</span>
    </div>
  );
};

// =============================================================================
// FULL PAGE SPINNER
// =============================================================================

export interface PageSpinnerProps {
  /** Show spinner */
  show?: boolean;
  /** Loading message */
  message?: string;
}

export const PageSpinner: React.FC<PageSpinnerProps> = ({
  show = true,
  message = 'Loading...',
}) => {
  if (!show) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-cream-50/80 backdrop-blur-sm">
      <Spinner size="xl" />
      {message && (
        <p className="mt-4 text-body-md text-navy-700">{message}</p>
      )}
    </div>
  );
};

export default Spinner;
