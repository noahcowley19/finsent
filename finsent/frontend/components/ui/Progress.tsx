'use client';

import React, { HTMLAttributes } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type ProgressSize = 'xs' | 'sm' | 'md' | 'lg';
export type ProgressVariant = 'default' | 'success' | 'warning' | 'error';

export interface ProgressProps extends HTMLAttributes<HTMLDivElement> {
  /** Progress value (0-100) */
  value: number;
  /** Maximum value */
  max?: number;
  /** Progress bar size */
  size?: ProgressSize;
  /** Color variant */
  variant?: ProgressVariant;
  /** Show percentage label */
  showLabel?: boolean;
  /** Show as indeterminate */
  indeterminate?: boolean;
  /** Animate progress changes */
  animated?: boolean;
}

// =============================================================================
// STYLES
// =============================================================================

const sizeStyles: Record<ProgressSize, string> = {
  xs: 'h-1',
  sm: 'h-1.5',
  md: 'h-2',
  lg: 'h-3',
};

const variantStyles: Record<ProgressVariant, string> = {
  default: 'bg-navy-500',
  success: 'bg-success-500',
  warning: 'bg-warning-500',
  error: 'bg-error-500',
};

const trackStyles = 'bg-cream-200';

// =============================================================================
// COMPONENT
// =============================================================================

export const Progress: React.FC<ProgressProps> = ({
  value,
  max = 100,
  size = 'md',
  variant = 'default',
  showLabel = false,
  indeterminate = false,
  animated = true,
  className = '',
  ...props
}) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));

  return (
    <div className={`w-full ${className}`} {...props}>
      {showLabel && !indeterminate && (
        <div className="flex justify-between items-center mb-1">
          <span className="text-body-sm text-neutral-600">Progress</span>
          <span className="text-body-sm font-medium text-navy-900">
            {Math.round(percentage)}%
          </span>
        </div>
      )}
      
      <div
        role="progressbar"
        aria-valuenow={indeterminate ? undefined : value}
        aria-valuemin={0}
        aria-valuemax={max}
        className={`
          w-full
          ${trackStyles}
          ${sizeStyles[size]}
          rounded-full
          overflow-hidden
        `}
      >
        <div
          className={`
            h-full
            ${variantStyles[variant]}
            rounded-full
            ${animated && !indeterminate ? 'transition-all duration-slow ease-out' : ''}
            ${indeterminate ? 'w-1/3 animate-[indeterminate_1.5s_ease-in-out_infinite]' : ''}
          `}
          style={indeterminate ? undefined : { width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

// =============================================================================
// CIRCULAR PROGRESS
// =============================================================================

export interface CircularProgressProps extends HTMLAttributes<HTMLDivElement> {
  /** Progress value (0-100) */
  value: number;
  /** Maximum value */
  max?: number;
  /** Circle size in pixels */
  size?: number;
  /** Stroke width in pixels */
  strokeWidth?: number;
  /** Color variant */
  variant?: ProgressVariant;
  /** Show percentage in center */
  showValue?: boolean;
  /** Show as indeterminate */
  indeterminate?: boolean;
}

export const CircularProgress: React.FC<CircularProgressProps> = ({
  value,
  max = 100,
  size = 48,
  strokeWidth = 4,
  variant = 'default',
  showValue = false,
  indeterminate = false,
  className = '',
  ...props
}) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percentage / 100) * circumference;

  const strokeColors: Record<ProgressVariant, string> = {
    default: '#254D70', // navy-500
    success: '#2D7A4F', // success-500
    warning: '#9A6B28', // warning-500
    error: '#B83A3A', // error-500
  };

  return (
    <div
      role="progressbar"
      aria-valuenow={indeterminate ? undefined : value}
      aria-valuemin={0}
      aria-valuemax={max}
      className={`relative inline-flex items-center justify-center ${className}`}
      style={{ width: size, height: size }}
      {...props}
    >
      <svg
        className={indeterminate ? 'animate-spin' : ''}
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
      >
        {/* Track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#E5D9C3" // cream-200
          strokeWidth={strokeWidth}
        />
        {/* Progress */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={strokeColors[variant]}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={indeterminate ? circumference * 0.75 : offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          className={indeterminate ? '' : 'transition-all duration-slow ease-out'}
        />
      </svg>
      
      {showValue && !indeterminate && (
        <span className="absolute text-body-sm font-medium text-navy-900">
          {Math.round(percentage)}%
        </span>
      )}
    </div>
  );
};

// =============================================================================
// USAGE INDICATOR (for feature limits)
// =============================================================================

export interface UsageIndicatorProps extends HTMLAttributes<HTMLDivElement> {
  /** Current usage count */
  used: number;
  /** Maximum allowed */
  total: number;
  /** Label text */
  label?: string;
  /** Show warning when near limit */
  warnAt?: number;
}

export const UsageIndicator: React.FC<UsageIndicatorProps> = ({
  used,
  total,
  label = 'Used',
  warnAt = 80,
  className = '',
  ...props
}) => {
  const percentage = (used / total) * 100;
  const isWarning = percentage >= warnAt && percentage < 100;
  const isError = percentage >= 100;

  const variant: ProgressVariant = isError ? 'error' : isWarning ? 'warning' : 'default';

  return (
    <div className={`w-full ${className}`} {...props}>
      <div className="flex justify-between items-center mb-1">
        <span className="text-body-sm text-neutral-600">{label}</span>
        <span
          className={`text-body-sm font-medium ${
            isError ? 'text-error-500' : isWarning ? 'text-warning-500' : 'text-navy-900'
          }`}
        >
          {used} of {total}
        </span>
      </div>
      <Progress value={used} max={total} variant={variant} size="sm" animated />
      {isError && (
        <p className="mt-1 text-caption text-error-500">
          Limit reached. Upgrade to continue.
        </p>
      )}
    </div>
  );
};

export default Progress;
