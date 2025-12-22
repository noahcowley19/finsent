'use client';

import React, { HTMLAttributes, ReactNode } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type BadgeVariant = 'success' | 'warning' | 'error' | 'info' | 'neutral';
export type BadgeSize = 'sm' | 'md';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  /** Color variant */
  variant?: BadgeVariant;
  /** Badge size */
  size?: BadgeSize;
  /** Show dot indicator */
  dot?: boolean;
  /** Icon to show */
  icon?: ReactNode;
  /** Content */
  children: ReactNode;
}

// =============================================================================
// STYLES
// =============================================================================

const baseStyles = `
  inline-flex items-center gap-1.5
  font-medium
  rounded-full
  whitespace-nowrap
`;

const variantStyles: Record<BadgeVariant, string> = {
  success: 'bg-success-100 text-success-700',
  warning: 'bg-warning-100 text-warning-700',
  error: 'bg-error-100 text-error-700',
  info: 'bg-navy-50 text-navy-700',
  neutral: 'bg-neutral-100 text-neutral-700',
};

const sizeStyles: Record<BadgeSize, string> = {
  sm: 'px-2 py-0.5 text-caption',
  md: 'px-2.5 py-1 text-body-sm',
};

const dotColors: Record<BadgeVariant, string> = {
  success: 'bg-success-500',
  warning: 'bg-warning-500',
  error: 'bg-error-500',
  info: 'bg-navy-500',
  neutral: 'bg-neutral-500',
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Badge: React.FC<BadgeProps> = ({
  variant = 'neutral',
  size = 'md',
  dot = false,
  icon,
  children,
  className = '',
  ...props
}) => {
  return (
    <span
      className={`
        ${baseStyles}
        ${variantStyles[variant]}
        ${sizeStyles[size]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {dot && (
        <span
          className={`w-1.5 h-1.5 rounded-full ${dotColors[variant]}`}
          aria-hidden="true"
        />
      )}
      {icon && (
        <span className={size === 'sm' ? 'w-3 h-3' : 'w-4 h-4'}>
          {icon}
        </span>
      )}
      {children}
    </span>
  );
};

// =============================================================================
// STATUS BADGE (with pulse animation for live status)
// =============================================================================

export interface StatusBadgeProps extends Omit<BadgeProps, 'dot' | 'icon'> {
  /** Show animated pulse */
  pulse?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  variant = 'success',
  pulse = false,
  children,
  className = '',
  ...props
}) => {
  return (
    <Badge variant={variant} className={className} {...props}>
      <span className="relative flex h-2 w-2">
        {pulse && (
          <span
            className={`absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping ${dotColors[variant]}`}
          />
        )}
        <span
          className={`relative inline-flex rounded-full h-2 w-2 ${dotColors[variant]}`}
        />
      </span>
      {children}
    </Badge>
  );
};

// =============================================================================
// COUNT BADGE (for notifications, etc.)
// =============================================================================

export interface CountBadgeProps extends Omit<BadgeProps, 'children'> {
  /** Count to display */
  count: number;
  /** Maximum count before showing "+" */
  max?: number;
  /** Show zero */
  showZero?: boolean;
}

export const CountBadge: React.FC<CountBadgeProps> = ({
  count,
  max = 99,
  showZero = false,
  variant = 'error',
  size = 'sm',
  className = '',
  ...props
}) => {
  if (count === 0 && !showZero) {
    return null;
  }

  const displayCount = count > max ? `${max}+` : count;

  return (
    <Badge
      variant={variant}
      size={size}
      className={`min-w-[1.25rem] justify-center ${className}`}
      {...props}
    >
      {displayCount}
    </Badge>
  );
};

export default Badge;
