'use client';

import React, { HTMLAttributes, ReactNode } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'accent';
export type BadgeSize = 'sm' | 'md';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  /** Visual variant */
  variant?: BadgeVariant;
  /** Size */
  size?: BadgeSize;
  /** Dot indicator */
  dot?: boolean;
  /** Content */
  children: ReactNode;
}

// =============================================================================
// STYLES - Modern SaaS Aesthetic
// =============================================================================

const baseStyles = 'inline-flex items-center font-medium rounded-full transition-colors';

const sizeStyles: Record<BadgeSize, string> = {
  sm: 'px-2 py-0.5 text-body-xs gap-1',
  md: 'px-2.5 py-1 text-body-sm gap-1.5',
};

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-ink-100 text-ink-700',
  success: 'bg-success-50 text-success-700',
  warning: 'bg-warning-50 text-warning-700',
  error: 'bg-error-50 text-error-700',
  accent: 'bg-accent/10 text-accent-700',
};

const dotStyles: Record<BadgeVariant, string> = {
  default: 'bg-ink-500',
  success: 'bg-success-500',
  warning: 'bg-warning-500',
  error: 'bg-error-500',
  accent: 'bg-accent',
};

// =============================================================================
// BADGE COMPONENT
// =============================================================================

export const Badge: React.FC<BadgeProps> = ({
  variant = 'default',
  size = 'sm',
  dot = false,
  children,
  className = '',
  ...props
}) => {
  return (
    <span
      className={`
        ${baseStyles}
        ${sizeStyles[size]}
        ${variantStyles[variant]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {dot && (
        <span className={`w-1.5 h-1.5 rounded-full ${dotStyles[variant]}`} />
      )}
      {children}
    </span>
  );
};

// =============================================================================
// STATUS BADGE - For showing status with animation
// =============================================================================

export interface StatusBadgeProps extends Omit<BadgeProps, 'dot'> {
  /** Pulse animation for active states */
  pulse?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  pulse = false,
  variant = 'default',
  children,
  className = '',
  ...props
}) => {
  return (
    <Badge
      variant={variant}
      className={className}
      {...props}
    >
      <span className="relative flex h-2 w-2">
        {pulse && (
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${dotStyles[variant]}`} />
        )}
        <span className={`relative inline-flex rounded-full h-2 w-2 ${dotStyles[variant]}`} />
      </span>
      {children}
    </Badge>
  );
};

// =============================================================================
// COUNT BADGE - For numeric counts
// =============================================================================

export interface CountBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  /** Count value */
  count: number;
  /** Maximum count to display */
  max?: number;
  /** Show zero */
  showZero?: boolean;
  /** Visual variant */
  variant?: BadgeVariant;
}

export const CountBadge: React.FC<CountBadgeProps> = ({
  count,
  max = 99,
  showZero = false,
  variant = 'default',
  className = '',
  ...props
}) => {
  if (count === 0 && !showZero) return null;

  const displayCount = count > max ? `${max}+` : count;

  return (
    <span
      className={`
        inline-flex items-center justify-center
        min-w-[1.25rem] h-5 px-1.5
        text-body-xs font-semibold
        rounded-full
        ${variantStyles[variant]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {displayCount}
    </span>
  );
};

export default Badge;
